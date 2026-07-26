// Package mcp exposes descry retrieval and graph queries to AI agents over the
// Model Context Protocol — JSON-RPC 2.0 spoken over stdio. An agent launches the
// descry binary as a subprocess and calls its tools; each tool is a thin wrapper
// over the search and graph layers.
//
// The tools are:
//   - search        — hybrid vector+BM25 search, returns ranked file locations
//   - read_relevant — search plus the matching source content, inline
//   - graph_impact  — transitive callers of a Go symbol (blast radius)
//   - graph_trace   — a call path between two Go symbols
//   - status        — which repository is served, and the state of every repo
//
// The package is built to need no configuration. Which repository a call
// operates on is resolved per call (see Server), repositories are opened lazily
// and indexed in the background, and no call ever blocks on indexing — so a
// single globally-registered server works across every project you open.
package mcp

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/jKiler/descry/internal/core"
	"github.com/jKiler/descry/internal/graph"
	"github.com/jKiler/descry/internal/skill"
)

const serverName = "descry"

// Workspace is one opened, searchable repository — everything the tools need to
// answer for a single root directory.
type Workspace struct {
	// Search returns ranked hybrid-search results for a query.
	Search func(query string, k int) []core.SearchResult
	// Chunks reports how many chunks are indexed.
	Chunks func() int
	// IndexPath is where the index database lives.
	IndexPath string
	// BuildGraph builds the call graph for graph_impact / graph_trace. It is
	// called at most once, lazily, the first time a graph tool is used — so a
	// search-only session never pays for it, and it stays off the path that
	// makes search available. May be nil if graphs are unsupported.
	BuildGraph func() (*graph.Graph, error)
	// Close releases the workspace's resources.
	Close func() error
}

// Phases reported by an Opener while it works, so a waiting call can say what is
// actually happening instead of always claiming to index. Until the Opener knows
// whether an index already exists (it must open the database to find out), the
// phase is the neutral phaseOpening.
const (
	phaseOpening  = "opening"  // opening the database; not yet known if an index exists
	phaseLoading  = "loading"  // opening an existing index (no rebuild)
	phaseIndexing = "indexing" // building the index for the first time
)

// Opener opens the workspace at root (an absolute path), building the index if
// there isn't one. It may be slow: the server always runs it on a background
// goroutine and reports progress through the progress callback (phase, plus
// done/total chunks while indexing).
type Opener func(root string, progress func(phase string, done, total int)) (*Workspace, error)

// rootsTimeout bounds the roots/list round trip. Clients that support roots
// answer in microseconds; clients that don't implement it may never answer at
// all, and must not be allowed to wedge a tool call.
const rootsTimeout = 2 * time.Second

// Server serves the descry tools over MCP.
//
// The repository a call operates on is resolved per call, in order:
//
//  1. the call's own root argument, matched against the client's roots;
//  2. Root, if set — pins every call to one repository;
//  3. the client's MCP roots (roots/list), which is what makes one
//     globally-registered server follow the project the editor has open;
//  4. the process working directory, as a last resort.
//
// Repositories are opened in the background and cached per root, so several
// projects can be served from one process and no call ever waits on indexing.
type Server struct {
	// Root, if non-empty, pins the server to one repository and skips roots
	// discovery.
	Root string
	// Open opens a workspace for a resolved root. Required.
	Open Opener

	mu      sync.Mutex
	repos   map[string]*repo
	noRoots map[string]bool // session ID -> client never answered roots/list
}

// repo tracks one repository's state. A repo is opened once, on a background
// goroutine; calls arriving meanwhile see progress instead of blocking. The call
// graph is a second, independent lazy stage (graphStarted…): only building it
// when a graph tool is first used keeps it off the path that makes search ready.
type repo struct {
	ws          *Workspace
	err         error
	ready       bool
	phase       string // phaseLoading | phaseIndexing
	done, total int    // embedding progress while indexing

	graphStarted bool
	graph        *graph.Graph
	graphErr     error
	graphReady   bool
}

// Serve runs the server over stdio until the client disconnects or ctx is
// cancelled.
func (s *Server) Serve(ctx context.Context, version string) error {
	return s.newServer(version).Run(ctx, &mcp.StdioTransport{})
}

// Close closes every workspace the server opened.
func (s *Server) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var firstErr error
	for _, r := range s.repos {
		if r.ws == nil || r.ws.Close == nil {
			continue
		}
		if err := r.ws.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	s.repos = nil
	return firstErr
}

// --- repository resolution ---

// Root sources, reported by the status tool so a surprising repository is
// self-diagnosing rather than a mystery.
const (
	sourceRequested = "requested"   // the call's root argument
	sourcePinned    = "pinned"      // explicit path (argument or DESCRY_ROOT)
	sourceClient    = "client-root" // the client's MCP roots
	sourceCWD       = "cwd"         // process working directory (last resort)
)

// resolution records which repository a call resolved to and why.
type resolution struct {
	Root       string
	Source     string
	Advertised []string // every file:// root the client reported, in order
}

// resolve picks the repository for this call. want is the call's optional root
// argument, matched leniently against the advertised roots so an agent can say
// "kubernetes" instead of an absolute path.
func (s *Server) resolve(ctx context.Context, sess *mcp.ServerSession, want string) resolution {
	advertised := s.clientRoots(ctx, sess)

	if want != "" {
		if m, ok := matchRoot(want, advertised); ok {
			return resolution{Root: m, Source: sourceRequested, Advertised: advertised}
		}
		// Not one of the advertised roots — take it as a path anyway, so an
		// explicit request always wins over guessing.
		return resolution{Root: want, Source: sourceRequested, Advertised: advertised}
	}
	if s.Root != "" {
		return resolution{Root: s.Root, Source: sourcePinned, Advertised: advertised}
	}
	if len(advertised) > 0 {
		return resolution{Root: advertised[0], Source: sourceClient, Advertised: advertised}
	}
	if cwd, err := os.Getwd(); err == nil {
		return resolution{Root: cwd, Source: sourceCWD}
	}
	return resolution{Root: ".", Source: sourceCWD}
}

// matchRoot resolves want against the advertised roots: an exact path, a
// basename ("kubernetes"), or any unambiguous substring. Reports false if
// nothing matches.
func matchRoot(want string, advertised []string) (string, bool) {
	if abs, err := filepath.Abs(want); err == nil {
		for _, r := range advertised {
			if r == abs {
				return r, true
			}
		}
	}
	want = strings.ToLower(want)
	for _, r := range advertised {
		if strings.ToLower(filepath.Base(r)) == want {
			return r, true
		}
	}
	for _, r := range advertised {
		if strings.Contains(strings.ToLower(r), want) {
			return r, true
		}
	}
	return "", false
}

// clientRoots asks the client which projects it currently has open, returning
// every usable file:// root in the order advertised. It is asked on every call
// (so switching projects is picked up immediately), but a client that fails or
// times out once is never asked again for that session — some clients don't
// implement roots/list and would otherwise cost rootsTimeout per call.
func (s *Server) clientRoots(ctx context.Context, sess *mcp.ServerSession) []string {
	if sess == nil {
		return nil
	}
	id := sess.ID()

	s.mu.Lock()
	skip := s.noRoots[id]
	s.mu.Unlock()
	if skip {
		return nil
	}

	ctx, cancel := context.WithTimeout(ctx, rootsTimeout)
	defer cancel()
	res, err := sess.ListRoots(ctx, nil)
	if err != nil {
		s.mu.Lock()
		if s.noRoots == nil {
			s.noRoots = make(map[string]bool)
		}
		s.noRoots[id] = true
		s.mu.Unlock()
		return nil
	}
	var roots []string
	for _, r := range res.Roots {
		p, ok := fileURIPath(r.URI)
		if !ok {
			continue
		}
		// Normalize exactly like acquire keys s.repos (filepath.Abs also
		// cleans), so repoStates' lookups can't miss over a trailing slash.
		if abs, err := filepath.Abs(p); err == nil {
			p = abs
		}
		roots = append(roots, p)
	}
	return roots
}

// fileURIPath converts a file:// URI to a local path.
func fileURIPath(uri string) (string, bool) {
	u, err := url.Parse(uri)
	if err != nil || u.Scheme != "file" || u.Path == "" {
		return "", false
	}
	return filepath.FromSlash(u.Path), true
}

// --- workspace acquisition (never blocks) ---

// acquire returns the workspace for this call's repository. If the repository
// isn't open yet it starts opening in the background and returns a nil
// workspace with a human-readable progress note — a cold repository must never
// stall a tool call, however long indexing takes.
func (s *Server) acquire(ctx context.Context, sess *mcp.ServerSession, want string) (resolution, *Workspace, string, error) {
	res := s.resolve(ctx, sess, want)
	abs, err := filepath.Abs(res.Root)
	if err != nil {
		return res, nil, "", fmt.Errorf("resolve repository root %q: %w", res.Root, err)
	}
	res.Root = abs

	s.mu.Lock()
	r, ok := s.repos[abs]
	if !ok {
		r = &repo{phase: phaseOpening}
		if s.repos == nil {
			s.repos = make(map[string]*repo)
		}
		s.repos[abs] = r
		go s.openAsync(abs, r)
	}
	ws, ready, openErr, phase, done, total := r.ws, r.ready, r.err, r.phase, r.done, r.total
	if ready && openErr != nil {
		// Don't cache the failure: a fixable cause (unwritable dir, transient
		// error) should be retried on the next call rather than poisoning the
		// session.
		delete(s.repos, abs)
	}
	s.mu.Unlock()

	switch {
	case !ready:
		return res, nil, openingNote(abs, phase, done, total), nil
	case openErr != nil:
		return res, nil, "", openErr
	default:
		return res, ws, "", nil
	}
}

// openAsync runs the (possibly slow) open on a background goroutine, recording
// progress so in-flight calls can report it.
func (s *Server) openAsync(root string, r *repo) {
	ws, err := s.Open(root, func(phase string, done, total int) {
		s.mu.Lock()
		r.phase, r.done, r.total = phase, done, total
		s.mu.Unlock()
	})
	s.mu.Lock()
	r.ws, r.err, r.ready = ws, err, true
	s.mu.Unlock()
}

// openingNote describes an in-progress open for a tool result, distinguishing a
// first-time index build from merely loading an existing index.
func openingNote(root, phase string, done, total int) string {
	switch phase {
	case phaseLoading:
		return fmt.Sprintf("Opening %s (loading its existing index). Retry this call in a moment.", root)
	case phaseIndexing:
		if total > 0 {
			return fmt.Sprintf("Indexing %s for the first time: %d/%d chunks embedded (%d%%). Retry this call in a few seconds.",
				root, done, total, done*100/total)
		}
		return fmt.Sprintf("Indexing %s for the first time (this repository has no index yet). Retry this call in a few seconds.", root)
	default: // phaseOpening — not yet known whether an index exists
		return fmt.Sprintf("Opening %s. Retry this call in a moment.", root)
	}
}

// graphFor returns the (cached) call graph for a ready workspace, building it on
// a background goroutine the first time it is asked for. A nil graph with a
// non-empty note means "still building — retry".
func (s *Server) graphFor(root string, w *Workspace) (*graph.Graph, string, error) {
	if w.BuildGraph == nil {
		return nil, "", errNoGraph
	}
	s.mu.Lock()
	r := s.repos[root]
	if r == nil { // the workspace came from acquire, so this should always exist
		s.mu.Unlock()
		return nil, "", errNoGraph
	}
	if !r.graphStarted {
		r.graphStarted = true
		go func() {
			g, err := w.BuildGraph()
			s.mu.Lock()
			r.graph, r.graphErr, r.graphReady = g, err, true
			s.mu.Unlock()
		}()
	}
	g, ready, gerr := r.graph, r.graphReady, r.graphErr
	s.mu.Unlock()

	switch {
	case !ready:
		return nil, fmt.Sprintf("Building the call graph for %s. Retry this call in a few seconds.", root), nil
	case gerr != nil:
		return nil, "", gerr
	case g == nil:
		return nil, "", errNoGraph
	default:
		return g, "", nil
	}
}

// --- tools ---

// searchToolDescription is the point-of-use surface an agent reads when picking
// a tool, so it carries the canonical skill.SearchWhen guidance — the skill
// body renders the same constant, and TestSearchGuidanceSharedWithSkill pins
// both surfaces to it.
const searchToolDescription = "Hybrid (vector + BM25) code search over the repository the client currently has open. " +
	skill.SearchWhen + " " +
	"Returns ranked matches as file paths with line ranges, best first, plus the repository they came from. " +
	"If several projects are open, pass root (a path, or just a directory name like \"kubernetes\") to pick one; call status to list them."

// newServer builds the MCP server and registers the tools.
func (s *Server) newServer(version string) *mcp.Server {
	srv := mcp.NewServer(&mcp.Implementation{Name: serverName, Version: version}, nil)

	mcp.AddTool(srv, &mcp.Tool{
		Name:        "search",
		Description: searchToolDescription,
	}, func(ctx context.Context, req *mcp.CallToolRequest, in searchInput) (*mcp.CallToolResult, searchOutput, error) {
		res, w, note, err := s.acquire(ctx, req.Session, in.Root)
		if err != nil {
			return nil, searchOutput{}, err
		}
		if w == nil {
			return nil, searchOutput{Root: res.Root, Indexing: true, Message: note}, nil
		}
		out := query(w, in, false)
		out.Root = res.Root
		return nil, out, nil
	})

	mcp.AddTool(srv, &mcp.Tool{
		Name: "read_relevant",
		Description: "Like search, but also returns the source content of each matching chunk so the code can be read inline without a separate file read. " +
			"Accepts the same optional root argument.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, in searchInput) (*mcp.CallToolResult, searchOutput, error) {
		res, w, note, err := s.acquire(ctx, req.Session, in.Root)
		if err != nil {
			return nil, searchOutput{}, err
		}
		if w == nil {
			return nil, searchOutput{Root: res.Root, Indexing: true, Message: note}, nil
		}
		out := query(w, in, true)
		out.Root = res.Root
		return nil, out, nil
	})

	mcp.AddTool(srv, &mcp.Tool{
		Name: "graph_impact",
		Description: "List the transitive callers of a Go symbol — the blast radius of changing it. " +
			"The symbol may be a bare name (\"termScore\") or a fully qualified id; if it is ambiguous, the candidates are returned instead.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, in symbolInput) (*mcp.CallToolResult, impactOutput, error) {
		res, w, note, err := s.acquire(ctx, req.Session, in.Root)
		if err != nil {
			return nil, impactOutput{}, err
		}
		if w == nil {
			return nil, impactOutput{Root: res.Root, Indexing: true, Message: note}, nil
		}
		g, gnote, gerr := s.graphFor(res.Root, w)
		if gerr != nil {
			return nil, impactOutput{}, gerr
		}
		if g == nil {
			return nil, impactOutput{Root: res.Root, Indexing: true, Message: gnote}, nil
		}
		id, matches, msg := resolveSymbol(g, in.Symbol)
		if id == "" {
			return nil, impactOutput{Root: res.Root, Matches: matches, Message: msg}, nil
		}
		return nil, impactOutput{Root: res.Root, Symbol: id, Callers: g.Impact(id)}, nil
	})

	mcp.AddTool(srv, &mcp.Tool{
		Name: "graph_trace",
		Description: "Find a call path from one Go symbol to another (forward reachability), or report that none exists. " +
			"Both symbols may be bare names; ambiguous ones return their candidates instead.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, in traceInput) (*mcp.CallToolResult, traceOutput, error) {
		res, w, note, err := s.acquire(ctx, req.Session, in.Root)
		if err != nil {
			return nil, traceOutput{}, err
		}
		if w == nil {
			return nil, traceOutput{Root: res.Root, Indexing: true, Message: note}, nil
		}
		g, gnote, gerr := s.graphFor(res.Root, w)
		if gerr != nil {
			return nil, traceOutput{}, gerr
		}
		if g == nil {
			return nil, traceOutput{Root: res.Root, Indexing: true, Message: gnote}, nil
		}
		from, matches, msg := resolveSymbol(g, in.From)
		if from == "" {
			return nil, traceOutput{Root: res.Root, Matches: matches, Message: "from: " + msg}, nil
		}
		to, matches, msg := resolveSymbol(g, in.To)
		if to == "" {
			return nil, traceOutput{Root: res.Root, Matches: matches, Message: "to: " + msg}, nil
		}
		path := g.Trace(from, to)
		return nil, traceOutput{Root: res.Root, From: from, To: to, Path: path, Found: path != nil}, nil
	})

	mcp.AddTool(srv, &mcp.Tool{
		Name: "status",
		Description: "Report which repository is being served and why, list every repository the client has open with its index state, " +
			"and show indexing progress. Call this first when results look like they came from the wrong project.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, in statusInput) (*mcp.CallToolResult, statusOutput, error) {
		res, w, note, err := s.acquire(ctx, req.Session, in.Root)
		if err != nil {
			return nil, statusOutput{}, err
		}
		out := statusOutput{
			Root:       res.Root,
			RootSource: res.Source,
			Available:  s.repoStates(res.Advertised),
		}
		if w == nil {
			out.Indexing, out.Message = true, note
			return nil, out, nil
		}
		out.Chunks, out.IndexPath = w.Chunks(), w.IndexPath
		return nil, out, nil
	})

	return srv
}

// repoStates describes each advertised repository's index state, so one status
// call explains the whole picture.
func (s *Server) repoStates(advertised []string) []repoState {
	s.mu.Lock()
	defer s.mu.Unlock()
	states := make([]repoState, 0, len(advertised))
	for _, root := range advertised {
		st := repoState{Root: root}
		if r, ok := s.repos[root]; ok {
			switch {
			case !r.ready:
				st.State = r.phase // "loading" | "indexing"
				if st.State == "" {
					st.State = "opening"
				}
				if r.total > 0 {
					st.Progress = fmt.Sprintf("%d%%", r.done*100/r.total)
				}
			case r.err != nil:
				st.State = "error"
				st.Error = r.err.Error()
			default:
				st.State = "ready"
				st.Chunks = r.ws.Chunks()
			}
		} else {
			st.State = "not opened"
		}
		states = append(states, st)
	}
	return states
}

// errNoGraph is returned by the graph tools when the call graph couldn't be
// built for the current repository (e.g. it contains no Go source).
var errNoGraph = fmt.Errorf("call graph unavailable for this repository (no parseable Go source?)")

// maxMatches caps an ambiguous symbol's candidate list — a loose substring match
// on a large repository can otherwise match thousands of nodes.
const maxMatches = 20

// resolveSymbol maps a caller-supplied symbol name onto a graph node. The typed
// resolver qualifies ids ("(*m/pkg.BM25).termScore"), so bare names must be
// resolved rather than looked up directly. It returns the resolved id, or the
// candidates plus an explanatory message when the name is unknown or ambiguous.
func resolveSymbol(g *graph.Graph, name string) (id string, matches []string, msg string) {
	ids := g.Resolve(name)
	switch len(ids) {
	case 0:
		return "", nil, fmt.Sprintf("no symbol matching %q in the call graph", name)
	case 1:
		return ids[0], nil, ""
	default:
		n := len(ids)
		if n > maxMatches {
			ids = ids[:maxMatches]
		}
		return "", ids, fmt.Sprintf("%q matches %d symbols; retry with one of the listed matches", name, n)
	}
}

const defaultK = 10

// query runs a search and maps the results into the tool output, optionally
// including each chunk's source content (read_relevant).
func query(w *Workspace, in searchInput, withContent bool) searchOutput {
	k := in.K
	if k <= 0 {
		k = defaultK
	}
	var out searchOutput
	for _, r := range w.Search(in.Query, k) {
		h := hit{
			Path:      r.Chunk.Path,
			StartLine: r.Chunk.StartLine,
			EndLine:   r.Chunk.EndLine,
			Symbol:    r.Chunk.Symbol,
			Score:     r.Score,
		}
		if withContent {
			h.Content = r.Chunk.Content
		}
		out.Hits = append(out.Hits, h)
	}
	return out
}

// --- tool input/output schemas (field docs become the JSON Schema) ---

// rootArg is the optional repository selector shared by every tool.
type rootArg struct {
	Root string `json:"root,omitempty" jsonschema:"optional repository to query when several are open: an absolute path or just a directory name (e.g. \"kubernetes\"); defaults to the project the client has open"`
}

type searchInput struct {
	rootArg
	Query string `json:"query" jsonschema:"the natural-language or keyword search query"`
	K     int    `json:"k,omitempty" jsonschema:"maximum number of results to return (default 10)"`
}

type hit struct {
	Path      string  `json:"path"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	Symbol    string  `json:"symbol,omitempty"`
	Score     float64 `json:"score"`
	Content   string  `json:"content,omitempty"`
}

type searchOutput struct {
	// Root is the repository these hits came from — echoed so a wrong-project
	// answer is obvious rather than silent.
	Root     string `json:"root"`
	Hits     []hit  `json:"hits"`
	Indexing bool   `json:"indexing,omitempty"`
	Message  string `json:"message,omitempty"`
}

type symbolInput struct {
	rootArg
	Symbol string `json:"symbol" jsonschema:"the function or method name to analyze"`
}

type impactOutput struct {
	Root string `json:"root"`
	// Symbol is the resolved node id the callers belong to — bare names are
	// resolved to qualified ids, so this shows what was actually analyzed.
	Symbol  string   `json:"symbol,omitempty"`
	Callers []string `json:"callers"`
	// Matches lists the candidates when the requested symbol was ambiguous.
	Matches  []string `json:"matches,omitempty"`
	Indexing bool     `json:"indexing,omitempty"`
	Message  string   `json:"message,omitempty"`
}

type traceInput struct {
	rootArg
	From string `json:"from" jsonschema:"the starting symbol"`
	To   string `json:"to" jsonschema:"the target symbol"`
}

type traceOutput struct {
	Root string `json:"root"`
	// From and To are the resolved node ids the path runs between.
	From  string   `json:"from,omitempty"`
	To    string   `json:"to,omitempty"`
	Path  []string `json:"path"`
	Found bool     `json:"found"`
	// Matches lists the candidates when one of the symbols was ambiguous.
	Matches  []string `json:"matches,omitempty"`
	Indexing bool     `json:"indexing,omitempty"`
	Message  string   `json:"message,omitempty"`
}

type statusInput struct {
	rootArg
}

// repoState is one repository's index state, as reported by status.
type repoState struct {
	Root     string `json:"root"`
	State    string `json:"state"` // ready | indexing | not opened | error
	Chunks   int    `json:"chunks,omitempty"`
	Progress string `json:"progress,omitempty"`
	Error    string `json:"error,omitempty"`
}

type statusOutput struct {
	Root string `json:"root"`
	// RootSource is why this repository was chosen: "requested", "pinned",
	// "client-root", or "cwd".
	RootSource string `json:"root_source"`
	Chunks     int    `json:"chunks,omitempty"`
	IndexPath  string `json:"index_path,omitempty"`
	Indexing   bool   `json:"indexing,omitempty"`
	Message    string `json:"message,omitempty"`
	// Available lists every repository the client has open, with its index state.
	Available []repoState `json:"available,omitempty"`
}
