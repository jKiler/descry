package main

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestParseArgs(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want invocation
	}{
		{"bare", nil, invocation{cmd: cmdHome}},
		{"query joins words", []string{"how", "is", "auth", "implemented"},
			invocation{cmd: cmdQuery, query: "how is auth implemented"}},
		{"quoted query", []string{"where is auth handled"},
			invocation{cmd: cmdQuery, query: "where is auth handled"}},
		{"subcommand", []string{"index", "some/dir"},
			invocation{cmd: "index", args: []string{"some/dir"}}},
		{"search is reserved", []string{"search", "index"},
			invocation{cmd: "search", args: []string{"index"}}},
		{"single reserved word is a subcommand, not a query", []string{"status"},
			invocation{cmd: "status", args: []string{}}},
		{"help word", []string{"help"}, invocation{cmd: cmdHelp}},
		{"help flag", []string{"--help"}, invocation{cmd: cmdHelp}},
		{"short help flag", []string{"-h"}, invocation{cmd: cmdHelp}},
		{"unknown flag is usage, not a query", []string{"--verbose", "auth"},
			invocation{cmd: cmdUsage}},
		{"blank query is usage", []string{"  "}, invocation{cmd: cmdUsage}},
		{"reserved word later in a query stays a query", []string{"how", "does", "index", "work"},
			invocation{cmd: cmdQuery, query: "how does index work"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := parseArgs(tc.args)
			// Normalize nil vs empty for comparison.
			if len(got.args) == 0 {
				got.args = nil
			}
			if len(tc.want.args) == 0 {
				tc.want.args = nil
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("parseArgs(%q) = %+v, want %+v", tc.args, got, tc.want)
			}
		})
	}
}

// Every name in the subcommands set must stay dispatchable, and the set is the
// single place that grows when a command is added — this test is the reminder
// that each addition is a breaking change for one-word queries (see the
// dispatch comment in main.go).
func TestReservedWordsDispatchAsSubcommands(t *testing.T) {
	for name := range subcommands {
		if got := parseArgs([]string{name}); got.cmd != name {
			t.Errorf("parseArgs(%q).cmd = %q, want the subcommand itself", name, got.cmd)
		}
	}
}

// The benchmark harness drives descry as an ordinary binary, so `descry search`
// has to accept the cutoff and spans-per-file a protocol specifies rather than
// only the two defaults tuned for a terminal.
func TestParseSearchArgsReadsFlagsAndKeepsTheRestAsQuery(t *testing.T) {
	dir, query, n, spans, err := parseSearchArgs([]string{"-n", "20", "-spans", "3", "how", "are", "globs", "compiled"})
	if err != nil {
		t.Fatal(err)
	}
	if n != 20 || spans != 3 {
		t.Errorf("n=%d spans=%d, want 20 and 3", n, spans)
	}
	if query != "how are globs compiled" {
		t.Errorf("query = %q", query)
	}
	if dir != "." {
		t.Errorf("dir = %q, want the working directory", dir)
	}
}

// Defaults must be exactly what the terminal form has always produced, or
// adding the flags would silently change what `descry search` shows.
func TestParseSearchArgsDefaultsMatchTheTerminalForm(t *testing.T) {
	_, query, n, spans, err := parseSearchArgs([]string{"parse", "a", "gitignore", "line"})
	if err != nil {
		t.Fatal(err)
	}
	if n != 10 || spans != searchSpansPerFile {
		t.Errorf("defaults n=%d spans=%d, want 10 and %d", n, spans, searchSpansPerFile)
	}
	if query != "parse a gitignore line" {
		t.Errorf("query = %q", query)
	}
}

func TestParseSearchArgsRejectsBadInput(t *testing.T) {
	for _, args := range [][]string{
		{},                   // no query
		{"-n"},               // flag with no value
		{"-n", "0", "q"},     // non-positive
		{"-spans", "x", "q"}, // not a number
	} {
		if _, _, _, _, err := parseSearchArgs(args); err == nil {
			t.Errorf("parseSearchArgs(%q) accepted bad input", args)
		}
	}
}

// The bare verb form must keep treating every argument as query text — a flag
// there would make one-word queries beginning with a dash unsearchable.
func TestBareVerbFormStillTreatsDashedArgsAsUsage(t *testing.T) {
	if got := parseArgs([]string{"-n"}).cmd; got != cmdUsage {
		t.Errorf("parseArgs([-n]) = %q, want usage — flags belong to the subcommand", got)
	}
}

// The default keeps the index inside the repository, where it travels with the
// code. Nothing about adding the override may change that.
func TestIndexPathDefaultsInsideTheRepository(t *testing.T) {
	t.Setenv(indexDirEnv, "")
	if got, want := dbPathFor("/repos/acme"), filepath.Join("/repos/acme", indexDirName, "index.db"); got != want {
		t.Errorf("dbPathFor = %q, want %q", got, want)
	}
}

// A benchmark scores pinned checkouts that must stay byte-identical, so the
// index has to be placeable outside the tree being read.
func TestIndexPathHonoursTheOverrideAndIsolatesRoots(t *testing.T) {
	out := t.TempDir()
	t.Setenv(indexDirEnv, out)

	a, b := dbPathFor("/repos/acme"), dbPathFor("/repos/other")
	for _, p := range []string{a, b} {
		if !strings.HasPrefix(p, out) {
			t.Errorf("index path %q escaped %q", p, out)
		}
		if strings.HasPrefix(p, "/repos/") {
			t.Errorf("index path %q is still inside the repository", p)
		}
	}
	// Two roots must not share a directory: an index whose fingerprint no longer
	// matches is cleared on open, so sharing would erase work on every swap.
	if filepath.Dir(a) == filepath.Dir(b) {
		t.Errorf("two roots share an index directory: %q", filepath.Dir(a))
	}
	// Stable for a given root, or every run would rebuild.
	if dbPathFor("/repos/acme") != a {
		t.Error("index path is not stable across calls")
	}
}

func TestSelectChunkerReadsTheEnvironment(t *testing.T) {
	t.Setenv(chunkerEnv, "")
	def, err := selectChunker()
	if err != nil {
		t.Fatalf("unset: %v", err)
	}
	t.Setenv(chunkerEnv, "ast-go")
	got, err := selectChunker()
	if err != nil {
		t.Fatalf("ast-go: %v", err)
	}
	if got.ID() == def.ID() {
		t.Fatalf("$%s did not change the chunker: both %q", chunkerEnv, got.ID())
	}
}

// The ID is what keeps two strategies' indexes apart. If a strategy stopped
// contributing a distinct one, a benchmark arm would silently reuse the other
// arm's chunks and report a difference of zero.
func TestChunkerIDsAreDistinct(t *testing.T) {
	seen := map[string]string{}
	for _, name := range []string{"", "ast-go", "line"} {
		t.Setenv(chunkerEnv, name)
		c, err := selectChunker()
		if err != nil {
			t.Fatalf("%q: %v", name, err)
		}
		if prev, dup := seen[c.ID()]; dup {
			t.Errorf("%q and %q share the fingerprint id %q", prev, name, c.ID())
		}
		seen[c.ID()] = name
	}
}

// An unknown name must fail rather than fall back: an arm that quietly measured
// the default would report no difference and read as a result.
func TestSelectChunkerRejectsAnUnknownName(t *testing.T) {
	t.Setenv(chunkerEnv, "definitely-not-a-chunker")
	if _, err := selectChunker(); err == nil {
		t.Fatal("unknown chunker accepted")
	} else if !strings.Contains(err.Error(), chunkerEnv) {
		t.Errorf("error does not name the variable at fault: %v", err)
	}
}

// `--` ends flag parsing, so a query beginning with a flag name is reachable.
// Without it the parser claims the word and then rejects the rest of the
// sentence as its value, and there is no way to search for the word at all.
func TestParseSearchArgsTreatsEverythingAfterDashDashAsQuery(t *testing.T) {
	_, query, n, _, err := parseSearchArgs([]string{"-n", "5", "--", "-n", "flag", "handling"})
	if err != nil {
		t.Fatalf("parseSearchArgs: %v", err)
	}
	if n != 5 {
		t.Errorf("flags before -- were not parsed: n=%d", n)
	}
	if query != "-n flag handling" {
		t.Errorf("query is %q, want %q", query, "-n flag handling")
	}
}

// A subcommand that takes an optional path must not treat a flag as one. descry
// *creates* the directory it is pointed at, so `descry index --verbose` used to
// silently make a directory called "--verbose" and write an index into it —
// leaving litter in the repository and reporting success.
func TestParseIndexArgsRejectsFlagsInsteadOfIndexingThem(t *testing.T) {
	for _, a := range []string{"--help", "--verbose", "-x", "-n", "--json=true"} {
		if dir, _, err := parseIndexArgs([]string{a}); err == nil {
			t.Errorf("descry index %s accepted, would index %q", a, dir)
		}
	}
	if _, _, err := parseIndexArgs([]string{"a", "b"}); err == nil {
		t.Error("two directories accepted; one would be silently discarded")
	}
}

func TestParseIndexArgsAcceptsTheValidForms(t *testing.T) {
	for _, tc := range []struct {
		args     []string
		wantDir  string
		wantJSON bool
	}{
		{nil, ".", false},
		{[]string{"-json"}, ".", true},
		{[]string{"--json"}, ".", true},
		{[]string{"/repo"}, "/repo", false},
		{[]string{"-json", "/repo"}, "/repo", true},
		{[]string{"/repo", "-json"}, "/repo", true},
	} {
		dir, asJSON, err := parseIndexArgs(tc.args)
		if err != nil {
			t.Errorf("%v: %v", tc.args, err)
			continue
		}
		if dir != tc.wantDir || asJSON != tc.wantJSON {
			t.Errorf("%v -> dir=%q json=%v, want dir=%q json=%v", tc.args, dir, asJSON, tc.wantDir, tc.wantJSON)
		}
	}
}

func TestIsFlagTreatsABareDashAsAPath(t *testing.T) {
	if isFlag("-") {
		t.Error(`"-" is conventionally stdin or a literal path, not a flag`)
	}
	if !isFlag("-x") || !isFlag("--long") {
		t.Error("leading-dash arguments should be flags")
	}
}
