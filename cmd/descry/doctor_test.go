package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/jKiler/descry/internal/chunk"
	"github.com/jKiler/descry/internal/store"
)

// pinDefaultModel forces the fp32 default for tests that compare embedder
// identity: staleFields reads DESCRY_MODEL from the ambient environment, so a
// developer with q8 exported would otherwise see spurious failures.
func pinDefaultModel(t *testing.T) {
	t.Helper()
	t.Setenv("DESCRY_MODEL", "")
}

// currentFingerprint is what an index built by this binary records, minus dim
// (which doctor deliberately doesn't compare — see staleFields).
func currentFingerprint(t *testing.T) string {
	t.Helper()
	return "schema=" + strconv.Itoa(store.SchemaVersion) +
		" pipeline=" + strconv.Itoa(pipelineVersion) +
		" embedder=all-MiniLM-L6-v2 dim=384 chunker=" + chunk.NewASTChunker().ID()
}

func TestParseFingerprint(t *testing.T) {
	got := parseFingerprint("schema=1 pipeline=3 embedder=all-MiniLM-L6-v2 dim=384 chunker=ast-go")
	for k, want := range map[string]string{
		"schema": "1", "pipeline": "3", "embedder": "all-MiniLM-L6-v2", "dim": "384", "chunker": "ast-go",
	} {
		if got[k] != want {
			t.Errorf("field %q = %q, want %q", k, got[k], want)
		}
	}
	if len(parseFingerprint("")) != 0 {
		t.Error("empty fingerprint should parse to no fields")
	}
}

// TestStaleFieldsMatchingIsQuiet pins that a current index reports nothing —
// doctor must not cry stale on a healthy repo.
func TestStaleFieldsMatchingIsQuiet(t *testing.T) {
	pinDefaultModel(t)
	if stale := staleFields(currentFingerprint(t)); len(stale) != 0 {
		t.Errorf("current fingerprint reported stale: %v", stale)
	}
	if stale := staleFields(""); len(stale) != 0 {
		t.Errorf("empty fingerprint reported stale: %v", stale)
	}
}

// TestStaleFieldsNamesTheDriftedField is the payoff of the whole check: an
// invalidated index should say *what* invalidated it, not merely that it is
// stale. Regression for the "silent rebuild, no explanation" behavior.
func TestStaleFieldsNamesTheDriftedField(t *testing.T) {
	pinDefaultModel(t)
	old := "schema=1 pipeline=0 embedder=all-MiniLM-L6-v2 dim=384 chunker=ast-go"
	stale := staleFields(old)
	if len(stale) == 0 {
		t.Fatal("an old pipeline version should report as stale")
	}
	joined := strings.Join(stale, ", ")
	if !strings.Contains(joined, "pipeline") {
		t.Errorf("stale report %q does not name the pipeline field", joined)
	}
	if strings.Contains(joined, "dim") {
		t.Errorf("stale report %q compares dim, which doctor cannot read without loading the model", joined)
	}
}

// TestStaleFieldsIgnoresUnknownFields keeps doctor forward-compatible: an index
// written by a newer descry carrying extra fingerprint fields must not produce
// bogus stale lines. This holds structurally (staleFields ranges over the
// fields it knows), so the test guards the property against a future rewrite
// that inverts the loop rather than catching a bug in today's code.
func TestStaleFieldsIgnoresUnknownFields(t *testing.T) {
	pinDefaultModel(t)
	fp := currentFingerprint(t) + " tokenizer=v9"
	if stale := staleFields(fp); len(stale) != 0 {
		t.Errorf("unknown fingerprint field reported stale: %v", stale)
	}
}

// TestDoctorIsAReservedSubcommand pins the breaking-change side of adding a
// verb: `descry doctor` must dispatch as a command, and the escape hatch for
// searching the word itself must still work.
func TestDoctorIsAReservedSubcommand(t *testing.T) {
	if got := parseArgs([]string{"doctor"}); got.cmd != "doctor" {
		t.Errorf("parseArgs([doctor]).cmd = %q, want doctor", got.cmd)
	}
	got := parseArgs([]string{"search", "doctor"})
	if got.cmd != "search" || len(got.args) != 1 || got.args[0] != "doctor" {
		t.Errorf("`descry search doctor` should still search for the word: %+v", got)
	}
}

// TestCheckORTBrokenOverrideIsBad pins the exit-code contract the README
// advertises: a DESCRY_ORT_LIB pointing at nothing is fatal, so `descry doctor`
// exits 1 and CI can gate on it. It must also name the variable, since that is
// the whole fix.
func TestCheckORTBrokenOverrideIsBad(t *testing.T) {
	t.Setenv("DESCRY_ORT_LIB", filepath.Join(t.TempDir(), "absent.dylib"))

	c := checkORT()
	if c.mark != markBad {
		t.Errorf("mark = %v, want markBad so doctor exits 1", c.mark)
	}
	if !strings.Contains(c.detail+c.hint, "DESCRY_ORT_LIB") {
		t.Errorf("report should name DESCRY_ORT_LIB: %q / %q", c.detail, c.hint)
	}
}

// TestCheckORTHonorsAWorkingOverride is the other half: a library that exists
// resolves cleanly, and doctor stays quiet.
func TestCheckORTHonorsAWorkingOverride(t *testing.T) {
	lib := filepath.Join(t.TempDir(), "libonnxruntime.dylib")
	if err := os.WriteFile(lib, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("DESCRY_ORT_LIB", lib)

	if c := checkORT(); c.mark != markOK {
		t.Errorf("mark = %v (%s), want markOK", c.mark, c.detail)
	}
}

// TestCheckIndexMissingIsAdvisory: a repository with no index is a normal cold
// start, not a broken install — doctor must not exit 1 on it, and must say how
// to build one.
func TestCheckIndexMissingIsAdvisory(t *testing.T) {
	checks := checkIndex(t.TempDir())
	if len(checks) == 0 {
		t.Fatal("checkIndex reported nothing for a cold repository")
	}
	if checks[0].mark == markBad {
		t.Errorf("mark = markBad for a cold repository; a missing index is advisory, not fatal")
	}
	if !strings.Contains(checks[0].hint, "descry index") {
		t.Errorf("hint should name `descry index`: %q", checks[0].hint)
	}
}

// TestDoctorChecksNeverPanic runs the whole report against a temp dir: cheap
// coverage that every check tolerates a machine with nothing set up.
func TestDoctorChecksNeverPanic(t *testing.T) {
	if got := doctorChecks(t.TempDir()); len(got) == 0 {
		t.Error("doctorChecks returned no checks")
	}
}

func TestHumanBytes(t *testing.T) {
	for _, tc := range []struct {
		in   int64
		want string
	}{
		{512, "512 B"},
		{1024, "1.0 KB"},
		{90 * 1024 * 1024, "90.0 MB"},
	} {
		if got := humanBytes(tc.in); got != tc.want {
			t.Errorf("humanBytes(%d) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
