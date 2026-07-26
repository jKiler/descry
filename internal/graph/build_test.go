package graph

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// BuildFromGoDir parses real .go files into a name-based call graph, and the
// graph methods work over it end to end. It must also ignore non-Go files and
// skip (not crash on) unparseable Go.
func TestBuildFromGoDir_CallGraph(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "a.go", `package sample

func Top() { Middle() }

func Middle() { Bottom() }

func Bottom() {}
`)
	writeFile(t, dir, "notes.md", "not go, must be ignored")
	writeFile(t, dir, "broken.go", "this is not valid go {{{")

	g, err := BuildFromGoDir(dir)
	if err != nil {
		t.Fatalf("BuildFromGoDir: %v", err)
	}

	if got := g.DependsOn("Top"); !reflect.DeepEqual(got, []string{"Middle"}) {
		t.Errorf("DependsOn(Top) = %v, want [Middle]", got)
	}
	if got := g.Dependents("Bottom"); !reflect.DeepEqual(got, []string{"Middle"}) {
		t.Errorf("Dependents(Bottom) = %v, want [Middle]", got)
	}
	if got := g.Impact("Bottom"); !reflect.DeepEqual(got, []string{"Middle", "Top"}) {
		t.Errorf("Impact(Bottom) = %v, want [Middle Top]", got)
	}
	if got := g.Trace("Top", "Bottom"); !reflect.DeepEqual(got, []string{"Top", "Middle", "Bottom"}) {
		t.Errorf("Trace(Top,Bottom) = %v, want [Top Middle Bottom]", got)
	}
}
