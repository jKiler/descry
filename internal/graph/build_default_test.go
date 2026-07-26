package graph

import (
	"os"
	"path/filepath"
	"testing"
)

// Build falls back to the name-based resolver when the typed one can't run —
// here, a directory with Go source but no go.mod (so packages.Load fails).
func TestBuildFallsBackToNamed(t *testing.T) {
	dir := t.TempDir()
	src := "package foo\n\nfunc A() { B() }\nfunc B() {}\n"
	if err := os.WriteFile(filepath.Join(dir, "foo.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	g, typed, err := Build(dir)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if typed {
		t.Error("expected fallback to the name-based resolver on a non-module dir")
	}
	if got := g.DependsOn("A"); len(got) != 1 || got[0] != "B" {
		t.Errorf("DependsOn(A) = %v, want [B]", got)
	}
}
