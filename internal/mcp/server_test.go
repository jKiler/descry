package mcp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/jKiler/descry/internal/core"
	"github.com/jKiler/descry/internal/graph"
	"github.com/jKiler/descry/internal/search"
)

// testWorkspace is a fixed in-memory workspace, so the server can be exercised
// end-to-end over the in-memory transport without an index or a model.
func testWorkspace(root string) *Workspace {
	g := graph.New()
	g.AddEdge("A", "B") // A -> B -> C
	g.AddEdge("B", "C")
	return &Workspace{
		// a.go carries two spans so the per-file arity is exercised; b.go one,
		// so a file with nothing more to show still comes back correctly.
		Search: func(_ string, _, perFile int) []search.FileHit {
			a := []core.Chunk{
				{Path: "a.go", StartLine: 1, EndLine: 3, Symbol: "Foo", Content: "func Foo() {}"},
				{Path: "a.go", StartLine: 20, EndLine: 24, Symbol: "Foo2", Content: "func Foo2() {}"},
			}
			return []search.FileHit{
				{Path: "a.go", Score: 0.9, Chunks: a[:min(perFile, len(a))]},
				{Path: "b.go", Score: 0.5, Chunks: []core.Chunk{
					{Path: "b.go", StartLine: 4, EndLine: 6, Symbol: "Bar", Content: "func Bar() {}"},
				}},
			}
		},
		Chunks:     func() int { return 7 },
		IndexPath:  filepath.Join(root, ".descry", "index.db"),
		BuildGraph: func() (*graph.Graph, error) { return g, nil },
		Close:      func() error { return nil },
	}
}

// instant returns an Opener that yields a ready workspace with no delay.
func instant() Opener {
	return func(root string, _ func(string, int, int)) (*Workspace, error) { return testWorkspace(root), nil }
}

// connect runs s on one end of an in-memory transport and returns a connected
// client session. clientRoots, if non-empty, are advertised as the client's MCP
// roots so repository resolution can be exercised.
func connect(t *testing.T, s *Server, clientRoots ...string) (context.Context, *mcp.ClientSession) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	serverT, clientT := mcp.NewInMemoryTransports()
	go s.newServer("test").Run(ctx, serverT) //nolint:errcheck // returns on ctx cancel

	client := mcp.NewClient(&mcp.Implementation{Name: "test-client", Version: "0"}, nil)
	for _, r := range clientRoots {
		client.AddRoots(&mcp.Root{URI: "file://" + r})
	}
	cs, err := client.Connect(ctx, clientT, nil)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(func() { cs.Close() })
	return ctx, cs
}

// callInto invokes a tool and decodes its structured output into out.
func callInto(t *testing.T, ctx context.Context, cs *mcp.ClientSession, name string, args map[string]any, out any) {
	t.Helper()
	res, err := cs.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		t.Fatalf("call %s: %v", name, err)
	}
	if res.IsError {
		t.Fatalf("call %s returned a tool error: %+v", name, res.Content)
	}
	b, err := json.Marshal(res.StructuredContent)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, out); err != nil {
		t.Fatalf("decode %s output: %v", name, err)
	}
}

// waitReady polls status until the repository has finished opening. Repositories
// open in the background, so every "happy path" test goes through this.
func waitReady(t *testing.T, ctx context.Context, cs *mcp.ClientSession, args map[string]any) statusOutput {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		var st statusOutput
		callInto(t, ctx, cs, "status", args, &st)
		if !st.Indexing {
			return st
		}
		if time.Now().After(deadline) {
			t.Fatalf("repository never became ready: %+v", st)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestSearchOmitsContent(t *testing.T) {
	root := t.TempDir()
	ctx, cs := connect(t, &Server{Root: root, Open: instant()})
	waitReady(t, ctx, cs, map[string]any{})

	var out searchOutput
	callInto(t, ctx, cs, "search", map[string]any{"query": "foo"}, &out)
	// Two files, and a.go contributes both of its spans at the default arity.
	if len(out.Hits) != 3 {
		t.Fatalf("hits = %d, want 3 (a.go twice, b.go once)", len(out.Hits))
	}
	if out.Root != root {
		t.Errorf("root = %q, want %q (results must name their repository)", out.Root, root)
	}
	if out.Hits[0].Path != "a.go" || out.Hits[0].Symbol != "Foo" {
		t.Errorf("first hit = %+v", out.Hits[0])
	}
	if out.Hits[0].Content != "" {
		t.Errorf("search must not include content, got %q", out.Hits[0].Content)
	}
}

func TestReadRelevantIncludesContent(t *testing.T) {
	ctx, cs := connect(t, &Server{Root: t.TempDir(), Open: instant()})
	waitReady(t, ctx, cs, map[string]any{})

	var out searchOutput
	callInto(t, ctx, cs, "read_relevant", map[string]any{"query": "foo"}, &out)
	if out.Hits[0].Content != "func Foo() {}" {
		t.Errorf("read_relevant content = %q, want the chunk source", out.Hits[0].Content)
	}
}

func TestGraphTools(t *testing.T) {
	ctx, cs := connect(t, &Server{Root: t.TempDir(), Open: instant()})
	waitReady(t, ctx, cs, map[string]any{})

	// The call graph builds lazily on first use, so poll until it's ready.
	var imp impactOutput
	deadline := time.Now().Add(5 * time.Second)
	for {
		imp = impactOutput{}
		callInto(t, ctx, cs, "graph_impact", map[string]any{"symbol": "C"}, &imp)
		if imp.Message == "" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("call graph never became ready: %+v", imp)
		}
		time.Sleep(5 * time.Millisecond)
	}
	if len(imp.Callers) != 2 { // A and B both reach C
		t.Fatalf("impact callers = %v, want [A B]", imp.Callers)
	}

	var tr traceOutput
	callInto(t, ctx, cs, "graph_trace", map[string]any{"from": "A", "to": "C"}, &tr)
	if !tr.Found || len(tr.Path) != 3 {
		t.Errorf("trace = %+v, want the path A->B->C", tr)
	}
}

// A cold repository must answer immediately with progress, never block.
func TestColdRepoReportsIndexingWithoutBlocking(t *testing.T) {
	release := make(chan struct{})
	s := &Server{Root: t.TempDir(), Open: func(root string, progress func(string, int, int)) (*Workspace, error) {
		progress("indexing", 5, 10)
		<-release // simulate a long index build
		return testWorkspace(root), nil
	}}
	ctx, cs := connect(t, s)

	done := make(chan searchOutput, 1)
	go func() {
		var out searchOutput
		callInto(t, ctx, cs, "search", map[string]any{"query": "foo"}, &out)
		done <- out
	}()

	select {
	case out := <-done:
		if !out.Indexing {
			t.Fatalf("cold repo should report indexing, got %+v", out)
		}
		if out.Message == "" {
			t.Error("indexing result should carry a human-readable message")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("search blocked on a cold repository instead of returning progress")
	}

	close(release)
	waitReady(t, ctx, cs, map[string]any{})

	var out searchOutput
	callInto(t, ctx, cs, "search", map[string]any{"query": "foo"}, &out)
	if out.Indexing || len(out.Hits) == 0 {
		t.Errorf("after indexing finished, search = %+v, want results", out)
	}
}

// A pinned root wins over whatever the client advertises.
func TestPinnedRootWins(t *testing.T) {
	pin := t.TempDir()
	ctx, cs := connect(t, &Server{Root: pin, Open: instant()}, t.TempDir())

	st := waitReady(t, ctx, cs, map[string]any{})
	if st.Root != pin {
		t.Errorf("root = %q, want the pinned %q", st.Root, pin)
	}
	if st.RootSource != sourcePinned {
		t.Errorf("root_source = %q, want %q", st.RootSource, sourcePinned)
	}
}

// With no pinned root, the server follows the client's advertised MCP root.
func TestFollowsClientRoot(t *testing.T) {
	workspaceRoot := t.TempDir()
	ctx, cs := connect(t, &Server{Open: instant()}, workspaceRoot)

	st := waitReady(t, ctx, cs, map[string]any{})
	if st.Root != workspaceRoot {
		t.Errorf("root = %q, want the client root %q", st.Root, workspaceRoot)
	}
	if st.RootSource != sourceClient {
		t.Errorf("root_source = %q, want %q", st.RootSource, sourceClient)
	}
}

// The root argument picks among several open projects — by basename, not just
// by absolute path.
func TestRootArgumentSelectsRepo(t *testing.T) {
	first, second := t.TempDir(), filepath.Join(t.TempDir(), "kubernetes")
	if err := mkdir(second); err != nil {
		t.Fatal(err)
	}
	ctx, cs := connect(t, &Server{Open: instant()}, first, second)

	args := map[string]any{"root": "kubernetes"}
	st := waitReady(t, ctx, cs, args)
	if st.Root != second {
		t.Errorf("root = %q, want %q selected by name", st.Root, second)
	}
	if st.RootSource != sourceRequested {
		t.Errorf("root_source = %q, want %q", st.RootSource, sourceRequested)
	}

	var out searchOutput
	callInto(t, ctx, cs, "search", map[string]any{"query": "foo", "root": "kubernetes"}, &out)
	if out.Root != second {
		t.Errorf("search answered from %q, want %q", out.Root, second)
	}
}

// status lists every open project and its index state.
func TestStatusListsAvailableRepos(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	ctx, cs := connect(t, &Server{Open: instant()}, a, b)

	st := waitReady(t, ctx, cs, map[string]any{})
	if len(st.Available) != 2 {
		t.Fatalf("available = %+v, want 2 repositories", st.Available)
	}
	var sawReady, sawUnopened bool
	for _, r := range st.Available {
		switch r.Root {
		case a:
			sawReady = r.State == "ready"
		case b:
			sawUnopened = r.State == "not opened" // never queried, so never opened
		}
	}
	if !sawReady {
		t.Errorf("expected the served repo %q to be ready: %+v", a, st.Available)
	}
	if !sawUnopened {
		t.Errorf("expected the unqueried repo %q to be listed as not opened: %+v", b, st.Available)
	}
}

// A workspace is opened once per root and reused across calls.
func TestWorkspaceCachedPerRoot(t *testing.T) {
	var opens atomic.Int32
	s := &Server{Root: t.TempDir(), Open: func(r string, _ func(string, int, int)) (*Workspace, error) {
		opens.Add(1)
		return testWorkspace(r), nil
	}}
	ctx, cs := connect(t, s)
	waitReady(t, ctx, cs, map[string]any{})

	var out searchOutput
	callInto(t, ctx, cs, "search", map[string]any{"query": "a"}, &out)
	callInto(t, ctx, cs, "search", map[string]any{"query": "b"}, &out)

	if got := opens.Load(); got != 1 {
		t.Errorf("workspace opened %d times, want 1 (cached)", got)
	}
}

func TestGraphToolsErrorWhenGraphNil(t *testing.T) {
	s := &Server{Root: t.TempDir(), Open: func(r string, _ func(string, int, int)) (*Workspace, error) {
		w := testWorkspace(r)
		w.BuildGraph = nil
		return w, nil
	}}
	ctx, cs := connect(t, s)
	waitReady(t, ctx, cs, map[string]any{})

	res, err := cs.CallTool(ctx, &mcp.CallToolParams{Name: "graph_impact", Arguments: map[string]any{"symbol": "C"}})
	if err == nil && !res.IsError {
		t.Error("graph_impact should report an error when the graph is unavailable")
	}
}

func TestMatchRoot(t *testing.T) {
	roots := []string{"/home/me/code/descry", "/home/me/code/kubernetes"}
	for _, tc := range []struct {
		want, expect string
		ok           bool
	}{
		{"kubernetes", "/home/me/code/kubernetes", true},
		{"DESCRY", "/home/me/code/descry", true}, // case-insensitive basename
		{"code/kubernetes", "/home/me/code/kubernetes", true},
		{"/home/me/code/descry", "/home/me/code/descry", true},
		{"nonesuch", "", false},
	} {
		got, ok := matchRoot(tc.want, roots)
		if ok != tc.ok || got != tc.expect {
			t.Errorf("matchRoot(%q) = (%q, %v), want (%q, %v)", tc.want, got, ok, tc.expect, tc.ok)
		}
	}
}

func TestIndexingNoteIncludesPercent(t *testing.T) {
	if note := openingNote("/repo", phaseIndexing, 5, 10); !strings.Contains(note, "50%") {
		t.Errorf("indexingNote = %q, want it to mention 50%%", note)
	}
}

// mkdir creates a directory for tests that need a specific basename.
func mkdir(p string) error { return os.MkdirAll(p, 0o755) }

// openingNote must distinguish loading an existing index from a first-time
// build — reporting "indexing" for an already-indexed repo was a reported bug.
func TestOpeningNoteLoadingVsIndexing(t *testing.T) {
	if note := openingNote("/repo", phaseLoading, 0, 0); strings.Contains(note, "Indexing") {
		t.Errorf("loading an existing index must not say 'Indexing': %q", note)
	}
	if note := openingNote("/repo", phaseLoading, 0, 0); !strings.Contains(note, "existing index") {
		t.Errorf("loading note should mention the existing index: %q", note)
	}
	if note := openingNote("/repo", phaseIndexing, 0, 0); !strings.Contains(note, "Indexing") {
		t.Errorf("first-time build should say 'Indexing': %q", note)
	}
}

// The graph builds lazily: a search-only session must never trigger it.
func TestGraphNotBuiltForSearchOnly(t *testing.T) {
	var graphBuilds atomic.Int32
	open := func(root string, _ func(string, int, int)) (*Workspace, error) {
		w := testWorkspace(root)
		w.BuildGraph = func() (*graph.Graph, error) {
			graphBuilds.Add(1)
			g := graph.New()
			g.AddEdge("A", "B")
			return g, nil
		}
		return w, nil
	}
	ctx, cs := connect(t, &Server{Root: t.TempDir(), Open: open})
	waitReady(t, ctx, cs, map[string]any{})

	var out searchOutput
	callInto(t, ctx, cs, "search", map[string]any{"query": "x"}, &out)
	callInto(t, ctx, cs, "status", map[string]any{}, &statusOutput{})

	if n := graphBuilds.Load(); n != 0 {
		t.Errorf("call graph was built %d times for a search-only session, want 0", n)
	}
}

// A file contributes several consecutive hits, best span first. This is the
// shipped answer to the measured failure mode — the right file
// ranked, the wrong part of it shown — so it needs a test that would fail if a
// refactor quietly collapsed a file back to one span.
func TestSearchReturnsSeveralSpansPerFile(t *testing.T) {
	ctx, cs := connect(t, &Server{Root: t.TempDir(), Open: instant()})
	waitReady(t, ctx, cs, map[string]any{})

	var out searchOutput
	callInto(t, ctx, cs, "search", map[string]any{"query": "foo"}, &out)

	var aSpans []hit
	for _, h := range out.Hits {
		if h.Path == "a.go" {
			aSpans = append(aSpans, h)
		}
	}
	if len(aSpans) != 2 {
		t.Fatalf("a.go returned %d spans, want 2", len(aSpans))
	}
	if aSpans[0].StartLine != 1 || aSpans[1].StartLine != 20 {
		t.Errorf("spans out of order or wrong: %+v", aSpans)
	}
	if aSpans[0].Score != aSpans[1].Score {
		t.Errorf("spans of one file carry different scores: %v vs %v",
			aSpans[0].Score, aSpans[1].Score)
	}
}

// chunks_per_file is the caller's context-budget dial, so it has to actually
// reach the workspace rather than being accepted and ignored.
func TestChunksPerFileIsHonoured(t *testing.T) {
	ctx, cs := connect(t, &Server{Root: t.TempDir(), Open: instant()})
	waitReady(t, ctx, cs, map[string]any{})

	for _, tc := range []struct{ perFile, wantA int }{{1, 1}, {2, 2}, {3, 2}} {
		var out searchOutput
		callInto(t, ctx, cs, "read_relevant",
			map[string]any{"query": "foo", "chunks_per_file": tc.perFile}, &out)
		got := 0
		for _, h := range out.Hits {
			if h.Path == "a.go" {
				got++
			}
		}
		if got != tc.wantA {
			t.Errorf("chunks_per_file=%d: a.go returned %d spans, want %d",
				tc.perFile, got, tc.wantA)
		}
	}
}
