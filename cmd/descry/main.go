// Command descry is the CLI front end: index / status / search / graph / eval.
// It persists the index to a local SQLite file so re-runs are instant.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/term"

	"github.com/jKiler/descry/internal/chunk"
	"github.com/jKiler/descry/internal/embed"
	"github.com/jKiler/descry/internal/eval"
	"github.com/jKiler/descry/internal/graph"
	"github.com/jKiler/descry/internal/index"
	"github.com/jKiler/descry/internal/mcp"
	"github.com/jKiler/descry/internal/search"
	"github.com/jKiler/descry/internal/skill"
	"github.com/jKiler/descry/internal/store"
)

// version is the descry release, reported over MCP to connecting clients.
// Release builds stamp the tag over it via -ldflags "-X main.version=…";
// this default covers source builds (`go build`, `go install`).
var version = "0.2.0"

func main() {
	inv := parseArgs(os.Args[1:])
	switch inv.cmd {
	case cmdHome:
		runHome(".")
	case cmdQuery:
		runQuery(".", inv.query)
	case "index":
		dir := "."
		if len(inv.args) > 0 {
			dir = inv.args[0]
		}
		runIndex(dir)
	case "status":
		runStatus(".")
	case "doctor":
		dir := "."
		if len(inv.args) > 0 {
			dir = inv.args[0]
		}
		runDoctor(dir)
	case "search":
		if len(inv.args) == 0 {
			fmt.Fprintln(os.Stderr, "usage: descry search <query>")
			os.Exit(2)
		}
		runSearch(".", strings.Join(inv.args, " "))
	case "graph":
		dir, mode := ".", "auto"
		for _, a := range inv.args {
			switch a {
			case "--typed":
				mode = "typed"
			case "--named":
				mode = "named"
			default:
				dir = a
			}
		}
		runGraph(dir, mode)
	case "eval":
		set := "eval_queries.json"
		if len(inv.args) > 0 {
			set = inv.args[0]
		}
		runEval(".", set)
	case "skill":
		sub := ""
		if len(inv.args) > 0 {
			sub = inv.args[0]
		}
		runSkill(sub)
	case "mcp":
		// An explicit root pins the server to one repository. Left empty, the
		// server follows whichever project the client has open (MCP roots).
		root := os.Getenv("DESCRY_ROOT")
		if len(inv.args) > 0 {
			root = inv.args[0]
		}
		runMCP(root)
	case cmdHelp:
		usage()
	default: // cmdUsage — an unknown flag or an empty query
		usage()
		os.Exit(2)
	}
}

// The verb-first dispatch: bare `descry` reports (or offers to build) the
// current repository's index, and any argument that isn't a subcommand is a
// search query — `descry how is auth implemented` just answers. The cost of
// that convenience is that subcommand names are reserved words: adding a new
// one is a breaking change for single-word queries, and `descry search <word>`
// is the escape hatch for querying a reserved word itself.
const (
	cmdHome  = "\x00home"  // bare `descry`
	cmdQuery = "\x00query" // free-text search
	cmdHelp  = "\x00help"
	cmdUsage = "\x00usage" // bad usage: unknown flag, empty query
)

// subcommands are the reserved first arguments that dispatch as commands
// rather than being read as a query.
var subcommands = map[string]bool{
	"index": true, "status": true, "search": true, "graph": true,
	"eval": true, "mcp": true, "skill": true, "doctor": true,
}

// invocation is one parsed command line.
type invocation struct {
	cmd   string   // a subcommand name, or one of the cmd* sentinels
	args  []string // the subcommand's remaining arguments
	query string   // the joined free-text query when cmd == cmdQuery
}

// parseArgs classifies the command line. Pure and total, so it is unit-testable
// apart from the side-effecting run* handlers.
func parseArgs(args []string) invocation {
	if len(args) == 0 {
		return invocation{cmd: cmdHome}
	}
	first := args[0]
	switch {
	case subcommands[first]:
		return invocation{cmd: first, args: args[1:]}
	case first == "help" || first == "-h" || first == "--help":
		return invocation{cmd: cmdHelp}
	case strings.HasPrefix(first, "-"):
		return invocation{cmd: cmdUsage}
	}
	query := strings.TrimSpace(strings.Join(args, " "))
	if query == "" {
		return invocation{cmd: cmdUsage}
	}
	return invocation{cmd: cmdQuery, query: query}
}

// indexDirName is the per-repository directory holding the index and the embed
// cache. It lives inside the repository being indexed, so the index travels with
// the code rather than with whatever directory a command happened to run in.
const indexDirName = ".descry"

// dbPathFor returns the index database path for a repository root.
func dbPathFor(root string) string { return filepath.Join(root, indexDirName, "index.db") }

// openPipeline opens the persistent SQLite index and wires it into a pipeline.
// The caller must Close the returned store.
// pipelineVersion is bumped whenever a change improves the QUALITY of stored data
// (smarter chunking, better tokenization, a new graph, etc.). Bumping it changes
// the index fingerprint, which transparently rebuilds every user's index on their
// next run. Record what each bump changed in the README provenance table.
const pipelineVersion = 3

func openPipeline(root string) (*index.Pipeline, *store.SQLiteStore, error) {
	dbPath := dbPathFor(root)
	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		return nil, nil, fmt.Errorf("create index dir: %w", err)
	}
	chk := chunk.NewASTChunker() // AST chunking, with a LineChunker fallback

	// Semantic embeddings via all-MiniLM-L6-v2 on ONNX Runtime. DESCRY_MODEL=q8
	// swaps in the int8-quantized export; its distinct embedder ID gives it its
	// own fingerprint and cache keyspace, so switching back and forth is a
	// (cache-warm) rebuild, never a corruption.
	modelPath, vocabPath, embedderID, err := embed.EnsureModel(os.Getenv("DESCRY_MODEL"))
	if err != nil {
		return nil, nil, err
	}
	onnx, err := embed.NewOrtEmbedderID(modelPath, vocabPath, embedderID)
	if err != nil {
		return nil, nil, err
	}
	// Persistent vector cache: fingerprint-driven rebuilds (pipeline bumps,
	// schema changes) re-embed only chunks whose text actually changed. Lives
	// beside the index but in its own file, so index clears never clear it.
	emb, err := embed.NewCachedEmbedder(onnx, filepath.Join(filepath.Dir(dbPath), "embed_cache.db"))
	if err != nil {
		return nil, nil, err
	}

	fp := store.Fingerprint{
		Schema:   store.SchemaVersion,
		Pipeline: pipelineVersion,
		Embedder: emb.ID(),
		Dim:      emb.Dim(),
		Chunker:  chk.ID(),
	}
	s, err := store.OpenSQLite(dbPath, fp)
	if err != nil {
		return nil, nil, err
	}
	if v, ok := envFloat("DESCRY_RERANK_MULT"); ok {
		s.SetRerankMult(int(v))
	}
	p := index.New(chk, emb, s)
	p.Progress = indexProgress()
	return p, s, nil
}

// indexProgress returns a Pipeline.Progress renderer: a \r-rewriting status line
// on stderr when it's a terminal, or periodic lines when piped (logs, CI).
// Every indexing phase reports, so a large repository never looks hung — the
// tree walk and the final commit each take real time before/after embedding.
//
// Rendered on stderr so `search` results on stdout stay clean. The callback is
// invoked concurrently from embed workers, hence the mutex; counts can arrive
// slightly out of order, hence the per-phase high-water-mark check.
func indexProgress() func(phase string, done, total int) {
	fi, err := os.Stderr.Stat()
	isTTY := err == nil && fi.Mode()&os.ModeCharDevice != 0

	var mu sync.Mutex
	var lastShown time.Time
	phase, best, lastStep := "", -1, -1
	return func(p string, done, total int) {
		mu.Lock()
		defer mu.Unlock()

		if p != phase { // entering a new phase: reset, and end the previous line
			if phase != "" && isTTY {
				fmt.Fprintln(os.Stderr)
			}
			phase, best, lastStep = p, -1, -1
		}
		if done < best {
			return // stale out-of-order update
		}
		best = done

		msg := phaseMessage(p, done, total)
		if isTTY {
			// Throttle repaints; always paint the start and the finish.
			if done != 0 && (total == 0 || done != total) && time.Since(lastShown) < 100*time.Millisecond {
				return
			}
			lastShown = time.Now()
			fmt.Fprintf(os.Stderr, "\r\033[K%s", msg)
			if total > 0 && done == total {
				fmt.Fprintln(os.Stderr)
				phase = "" // finished cleanly; don't emit a second newline
			}
			return
		}
		// Piped: one line per 10% (or per 500 files while scanning), so logs
		// stay readable but still show life.
		step := done
		if total > 0 {
			step = done * 10 / total
		} else {
			step = done / 500
		}
		if step != lastStep {
			lastStep = step
			fmt.Fprintln(os.Stderr, msg)
		}
	}
}

// phaseMessage renders one progress line for a phase.
func phaseMessage(phase string, done, total int) string {
	switch phase {
	case index.PhaseScanning:
		if total > 0 && done >= total {
			return fmt.Sprintf("scanned %d files", total)
		}
		return fmt.Sprintf("scanning %d files", done)
	case index.PhaseEmbedding:
		if total > 0 {
			return fmt.Sprintf("embedding %d/%d chunks (%3d%%)", done, total, done*100/total)
		}
		return "embedding chunks"
	case index.PhaseStoring:
		if done >= total && total > 0 {
			return fmt.Sprintf("stored %d chunks", total)
		}
		return fmt.Sprintf("writing %d chunks to the index", total)
	default:
		return phase
	}
}

// buildHybrid constructs the hybrid retriever, letting the RRF weights be
// overridden at runtime via DESCRY_VEC_WEIGHT / DESCRY_LEX_WEIGHT. This is what
// makes weight sweeps cheap — no recompile: build once, re-run `descry eval`
// with different env values against the same persisted index.
func buildHybrid(p *index.Pipeline, lex *search.BM25) *search.Hybrid {
	h := search.NewHybrid(p.Emb, p.Store, lex)
	if v, ok := envFloat("DESCRY_VEC_WEIGHT"); ok {
		h.VecWeight = v
	}
	if v, ok := envFloat("DESCRY_LEX_WEIGHT"); ok {
		h.LexWeight = v
	}
	if v, ok := envFloat("DESCRY_RRF_K"); ok {
		h.RRFK = v
	}
	if v, ok := envFloat("DESCRY_CAND_MULT"); ok {
		h.CandMult = int(v)
	}
	if v, ok := envFloat("DESCRY_LEXFILE_WEIGHT"); ok {
		h.LexFileWeight = v
	}
	if v, ok := envFloat("DESCRY_FUSE_ALPHA"); ok {
		h.FuseAlpha = v
	}
	return h
}

// newBM25 builds the lexical ranker, honoring DESCRY_BM25_PATH=0 to disable
// path/symbol tokens (an A/B switch for eval; the default is on).
func newBM25() *search.BM25 {
	lex := search.NewBM25()
	if os.Getenv("DESCRY_BM25_PATH") == "0" {
		lex.PathTokens = false
	}
	return lex
}

// loadOrBuildBM25 returns one of the two lexical rankers (kind selects
// store.LexicalChunks or store.LexicalFiles) over the store's chunks. It loads
// the persisted inverted index when present — avoiding the from-scratch rebuild
// that dominates startup on a large repo — and otherwise builds and persists it.
// The persisted index is the canonical, default-config build; when a build-time
// override is active (DESCRY_BM25_PATH=0) it builds a fresh in-memory index and
// leaves the cache untouched.
func loadOrBuildBM25(s *store.SQLiteStore, kind int) *search.BM25 {
	lex := newBM25()
	chunks := s.All()
	docs := chunks
	if kind == store.LexicalFiles {
		docs = search.FileDocs(chunks)
	}
	canonical := lex.PathTokens // the only build-time config knob today

	if canonical {
		if data, n, ok := s.LoadLexical(kind); ok && n == len(chunks) {
			if b, ok := search.DecodeBM25(data, docs); ok {
				return b
			}
		}
	}
	lex.Index(docs)
	if canonical {
		if data := lex.Encode(); len(data) > 0 {
			_ = s.SaveLexical(kind, data, len(chunks)) // best effort; a miss just rebuilds next time
		}
	}
	return lex
}

// readyHybrid makes dir searchable: it indexes it if the store is empty (later
// runs reuse the persisted index), then builds both lexical rankers (chunk- and
// file-level) and the hybrid retriever over them.
func readyHybrid(p *index.Pipeline, s *store.SQLiteStore, dir string) (*search.Hybrid, error) {
	if s.Len() == 0 {
		if _, err := index.IndexDir(p, dir); err != nil {
			return nil, err
		}
	}
	h := buildHybrid(p, loadOrBuildBM25(s, store.LexicalChunks))
	h.LexFile = loadOrBuildBM25(s, store.LexicalFiles)
	return h, nil
}

// envFloat reads a float64 from an env var; ok is false if unset or unparseable.
func envFloat(key string) (float64, bool) {
	s := os.Getenv(key)
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// runIndex indexes dir, writing the index into dir/.descry.
func runIndex(dir string) {
	p, s, err := openPipeline(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open error:", err)
		os.Exit(1)
	}
	defer s.Close()

	if s.Len() > 0 {
		// Name index.db specifically: deleting the whole .descry dir also
		// deletes embed_cache.db, which is what makes a rebuild nearly free.
		fmt.Printf("index already has %d chunks in %s\n", s.Len(), dbPathFor(dir))
		fmt.Printf("to rebuild, delete that index.db file — keep embed_cache.db beside it, its cached vectors make the rebuild fast\n")
		return
	}
	files, err := index.IndexDir(p, dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "index error:", err)
		os.Exit(1)
	}
	// Building + persisting the lexical indexes is seconds on a large repo, and
	// silent otherwise.
	fmt.Fprintln(os.Stderr, "building lexical indexes")
	loadOrBuildBM25(s, store.LexicalChunks) // do both now, so the first search is fast
	loadOrBuildBM25(s, store.LexicalFiles)
	fmt.Printf("indexed %d files, %d chunks into %s\n", files, s.Len(), dbPathFor(dir))
	nudgeSkill()
}

// nudgeSkill keeps the user-level agent skill in play after an index: a stale
// installed copy is silently refreshed (the user opted in by installing it),
// and a missing one earns a one-line tip. Best-effort — never fails the index.
func nudgeSkill() {
	switch installed, stale := skill.UserState(); {
	case installed && stale:
		if _, err := skill.InstallUser(); err == nil {
			fmt.Fprintln(os.Stderr, "refreshed the descry agent skill (user level)")
		}
	case !installed:
		fmt.Fprintln(os.Stderr, "tip: `descry skill install` teaches coding agents (Claude Code, Codex, …) to search with descry")
	}
}

// runSkill prints the agent skill ("descry skill") or installs it at user
// level ("descry skill install").
func runSkill(sub string) {
	switch sub {
	case "":
		fmt.Print(skill.Markdown())
	case "install":
		written, err := skill.InstallUser()
		if err != nil {
			fmt.Fprintln(os.Stderr, "skill install error:", err)
			os.Exit(1)
		}
		home, _ := os.UserHomeDir()
		for _, rel := range written {
			fmt.Printf("installed %s\n", filepath.Join(home, rel))
		}
		fmt.Println("agents can now load the /descry skill; re-run after upgrading descry to refresh it")
	default:
		fmt.Fprintf(os.Stderr, "unknown skill subcommand %q\nusage: descry skill [install]\n", sub)
		os.Exit(2)
	}
}

// runHome is bare `descry`: a one-screen home view for a warm repository, and
// an offer to index a cold one. It never indexes without consent — on a
// non-interactive stdin it only reports and hints.
func runHome(dir string) {
	if !indexExists(dir) {
		if !stdinIsTTY() {
			fmt.Printf("no index at %s — run `descry index` to build one\n", dbPathFor(dir))
			return
		}
		if !confirmIndex(dir) {
			fmt.Fprintln(os.Stderr, "not indexing — run `descry index` when ready")
			return
		}
		runIndex(dir)
		return
	}

	p, s, err := openPipeline(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open error:", err)
		os.Exit(1)
	}
	defer s.Close()
	fmt.Printf("%d chunks indexed in %s\n", s.Len(), dbPathFor(dir))
	fmt.Printf("embedder: %s\n", p.Emb.ID())
	fmt.Printf("\ntry: descry \"where is auth handled\"\n")
}

// runQuery is the verb itself: `descry <anything not a subcommand>` searches.
// A cold repository asks for consent first (TTY) or explains how to index
// (non-interactive), so a query can never silently start a long index build.
func runQuery(dir, query string) {
	if !indexExists(dir) {
		if !stdinIsTTY() {
			fmt.Fprintf(os.Stderr, "no index at %s — run `descry index` first\n", dbPathFor(dir))
			os.Exit(1)
		}
		if !confirmIndex(dir) {
			fmt.Fprintln(os.Stderr, "not indexing — run `descry index` when ready")
			os.Exit(1)
		}
		// Consent given: fall through — readyHybrid builds the index, then the
		// same run answers the query.
	}
	runSearch(dir, query)
}

// indexExists reports whether dir already has an index database. Checked
// before openPipeline on the consent paths because opening creates the
// .descry directory and an empty database as a side effect.
func indexExists(dir string) bool {
	_, err := os.Stat(dbPathFor(dir))
	return err == nil
}

// stdinIsTTY reports whether stdin is an interactive terminal — the gate for
// ever prompting. Pipes, redirects, and agent harnesses must get plain output,
// never a hanging question. This must be a real isatty check: a ModeCharDevice
// test wrongly passes for /dev/null, which would hang the prompt on a script's
// closed stdin.
func stdinIsTTY() bool {
	return term.IsTerminal(int(os.Stdin.Fd()))
}

// confirmIndex asks, on the terminal, whether to index dir. The prompt names
// the absolute directory so it is always clear what is about to be walked —
// the guard against accidentally indexing a home directory. Default is yes.
func confirmIndex(dir string) bool {
	abs, err := filepath.Abs(dir)
	if err != nil {
		abs = dir
	}
	fmt.Fprintf(os.Stderr, "descry: no index in %s — index it now? [Y/n] ", abs)
	var answer string
	if _, err := fmt.Fscanln(os.Stdin, &answer); err != nil && err.Error() != "unexpected newline" {
		return false // EOF or a read error is a "no", never a default-yes
	}
	switch strings.ToLower(strings.TrimSpace(answer)) {
	case "", "y", "yes":
		return true
	default:
		return false
	}
}

func runStatus(root string) {
	dbPath := dbPathFor(root)
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		fmt.Printf("no index at %s — run `descry index` first\n", dbPath)
		return
	}
	_, s, err := openPipeline(root)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open error:", err)
		os.Exit(1)
	}
	defer s.Close()
	fmt.Printf("%d chunks indexed in %s\n", s.Len(), dbPath)
}

func runSearch(dir, query string) {
	p, s, err := openPipeline(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open error:", err)
		os.Exit(1)
	}
	defer s.Close()

	h, err := readyHybrid(p, s, dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "index error:", err)
		os.Exit(1)
	}

	// File-level results: one hit per file (its best chunk), which is both what
	// the eval validates and what a caller scanning results wants.
	results := h.SearchFiles(query, 10)
	if len(results) == 0 {
		fmt.Println("no results")
		return
	}
	for i, r := range results {
		loc := fmt.Sprintf("%s:%d-%d", r.Chunk.Path, r.Chunk.StartLine, r.Chunk.EndLine)
		fmt.Printf("%2d. [%.3f] %s\n", i+1, r.Score, loc)
	}
}

func runEval(root, querySetPath string) {
	p, s, err := openPipeline(root)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open error:", err)
		os.Exit(1)
	}
	defer s.Close()

	h, err := readyHybrid(p, s, root)
	if err != nil {
		fmt.Fprintln(os.Stderr, "index error:", err)
		os.Exit(1)
	}

	// Gold labels are file paths, so retrieve at file granularity: each ranker's
	// chunk list collapses to files before fusion (see Hybrid.SearchFiles).
	// (Summing RRF scores per file was tried and is much worse — chunk-rich
	// files crowd out the single best hit.)
	retrieve := func(query string, k int) []string {
		results := h.SearchFiles(query, k)
		paths := make([]string, len(results))
		for i, r := range results {
			paths[i] = r.Chunk.Path
		}
		return paths
	}

	set, err := loadQuerySet(querySetPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "queryset error:", err)
		os.Exit(1)
	}
	if os.Getenv("DESCRY_EVAL_VERBOSE") == "1" {
		rep, perQuery := eval.EvaluateVerbose(retrieve, set, 10)
		for _, qr := range perQuery {
			rank := fmt.Sprintf("%4d", qr.Rank)
			if qr.Rank == 0 {
				rank = "MISS"
			}
			fmt.Printf("%s  %s\n", rank, qr.Query)
		}
		fmt.Printf("queries=%d  Recall@%d=%.3f  MRR=%.3f\n", rep.Queries, rep.K, rep.RecallAtK, rep.MRR)
		return
	}

	// Default report: recall at several cutoffs plus MRR, one line.
	rep := eval.EvaluateAtKs(retrieve, set, []int{5, 7, 10, 15, 20})
	fmt.Printf("queries=%d ", rep.Queries)
	for i, k := range rep.Ks {
		fmt.Printf(" R@%d=%.1f%%", k, rep.Recall[i]*100)
	}
	fmt.Printf("  MRR=%.3f\n", rep.MRR)
}

func loadQuerySet(path string) ([]eval.Labeled, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var set []eval.Labeled
	if err := json.Unmarshal(data, &set); err != nil {
		return nil, err
	}
	return set, nil
}

// runGraph prints the call graph as Mermaid. mode selects the resolver:
// "typed" (go/types, strict), "named" (name-based heuristic), or "auto"
// (default: typed with an automatic fallback to name-based).
func runGraph(dir, mode string) {
	var (
		g   *graph.Graph
		err error
	)
	switch mode {
	case "typed":
		g, err = graph.BuildFromGoDirTyped(dir) // strict; dir must be a module
	case "named":
		g, err = graph.BuildFromGoDir(dir) // name-based heuristic
	default: // auto
		var typed bool
		g, typed, err = graph.Build(dir)
		if err == nil && !typed {
			fmt.Fprintln(os.Stderr, "note: typed graph unavailable, used the name-based resolver")
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "graph error:", err)
		os.Exit(1)
	}
	fmt.Println(g.ToMermaid()) // paste into any Mermaid renderer
}

// runMCP serves descry over the Model Context Protocol on stdio.
//
// If root is empty the server follows whichever project the client has open
// (via MCP roots), so one globally-configured server works across repositories.
// A non-empty root pins it to that repository. Repositories are opened lazily on
// first use — an MCP client launches the server from an arbitrary working
// directory, so nothing may be opened relative to the process CWD at startup.
func runMCP(root string) {
	if root != "" {
		abs, err := filepath.Abs(root)
		if err != nil {
			fmt.Fprintln(os.Stderr, "resolve root:", err)
			os.Exit(1)
		}
		if fi, statErr := os.Stat(abs); statErr != nil || !fi.IsDir() {
			fmt.Fprintf(os.Stderr, "descry mcp: %q is not a directory.\n"+
				"Pass a repository path, e.g. `descry mcp /path/to/repo` (or set DESCRY_ROOT),\n"+
				"or omit it to follow the project your MCP client has open.\n", abs)
			os.Exit(1)
		}
		root = abs
	}

	srv := &mcp.Server{Root: root, Open: openWorkspace}
	defer srv.Close()

	if err := srv.Serve(context.Background(), version); err != nil {
		fmt.Fprintln(os.Stderr, "mcp error:", err)
		os.Exit(1)
	}
}

// openWorkspace opens one repository for the MCP server: the index (building it
// if absent) and the hybrid retriever. The MCP server runs this on a background
// goroutine and forwards progress to in-flight tool calls, so a cold repository
// never blocks a request. The call graph is deliberately NOT built here — it is
// built lazily via BuildGraph on first graph-tool use, so a search-only session
// stays fast. Diagnostics go to stderr so stdout stays reserved for JSON-RPC.
func openWorkspace(root string, progress func(phase string, done, total int)) (*mcp.Workspace, error) {
	p, s, err := openPipeline(root)
	if err != nil {
		return nil, fmt.Errorf("open index for %s (descry writes to <root>/%s, which must be writable): %w",
			root, indexDirName, err)
	}

	if s.Len() == 0 { // readyHybrid will build the index; announce and wire progress
		fmt.Fprintf(os.Stderr, "descry: no index at %s — building it now\n", dbPathFor(root))
		progress("indexing", 0, 0)
		// Report embedding progress to the MCP caller as well as stderr.
		stderrProgress := p.Progress
		p.Progress = func(phase string, done, total int) {
			if stderrProgress != nil {
				stderrProgress(phase, done, total)
			}
			// The MCP side only distinguishes "indexing"; the sub-phase detail
			// goes to stderr (the client's log) via stderrProgress.
			progress("indexing", done, total)
		}
	} else {
		fmt.Fprintf(os.Stderr, "descry: loading existing index for %s (%d chunks)\n", root, s.Len())
		progress("loading", 0, 0)
	}

	h, err := readyHybrid(p, s, root)
	if err != nil {
		s.Close()
		return nil, fmt.Errorf("index %s: %w", root, err)
	}
	fmt.Fprintf(os.Stderr, "descry: serving %s (%d chunks)\n", root, s.Len())

	return &mcp.Workspace{
		Search:    h.SearchFiles, // one hit per file — the benchmarked retrieval surface
		Chunks:    s.Len,
		IndexPath: dbPathFor(root),
		BuildGraph: func() (*graph.Graph, error) {
			// Typed (go/types), falling back to name-based. Built once, lazily.
			g, typed, gerr := graph.Build(root)
			if gerr != nil {
				return nil, gerr
			}
			resolver := "name-based"
			if typed {
				resolver = "typed"
			}
			fmt.Fprintf(os.Stderr, "descry: %s call graph for %s (%d symbols)\n", resolver, root, len(g.Nodes()))
			return g, nil
		},
		Close: s.Close,
	}, nil
}

func usage() {
	fmt.Println(`descry — a local hybrid (vector + BM25) code search index

usage:
  descry                 this repo's index status; offers to build a missing index
  descry <query>         search this repo (quotes optional); a cold repo asks first
  descry index [dir]     walk dir, chunk + embed + store (default ".")
  descry status          report how many chunks are indexed
  descry doctor [dir]    health-check the runtime, model, index and agent skill
                         (read-only; exits 1 if something is broken)
  descry search <query>  explicit search; indexes a cold repo without asking (script-friendly)
  descry graph [dir] [--typed|--named]  print the call graph as Mermaid (default: typed, falls back to name-based)
  descry eval [queryset.json]   score retrieval (Recall@k, MRR) vs a labeled set
  descry mcp [dir]       serve over MCP (JSON-RPC on stdio); without dir, follows
                         the project your MCP client has open (also DESCRY_ROOT)
  descry skill [install] print the agent skill, or install it for coding agents
                         (user level: ~/.claude/skills and ~/.agents/skills)`)
}
