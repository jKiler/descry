package graph

import (
	"strings"
	"testing"
)

func findNodeBySuffix(g *Graph, suffix string) string {
	for _, n := range g.Nodes() {
		if strings.HasSuffix(n, suffix) {
			return n
		}
	}
	return ""
}

// The whole point of the typed graph: two methods named Do on DIFFERENT types
// must resolve to distinct nodes. The name heuristic collapses them into one.
//
// This test type-checks a fixture module, so it spawns the go toolchain via
// packages.Load and is heavier than the other graph tests.
func TestBuildFromGoDirTyped_DistinguishesSameNamedMethods(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module m\n\ngo 1.21\n")
	writeFile(t, dir, "code.go", `package m

type A struct{}

func (A) Do() {}

type B struct{}

func (B) Do() {}

func Run() {
	var a A
	var b B
	a.Do()
	b.Do()
}
`)

	g, err := BuildFromGoDirTyped(dir)
	if err != nil {
		t.Fatalf("BuildFromGoDirTyped: %v", err)
	}

	run := findNodeBySuffix(g, ".Run")
	if run == "" {
		t.Fatalf("no Run caller node found; nodes = %v", g.Nodes())
	}

	deps := g.DependsOn(run)
	if len(deps) != 2 {
		t.Fatalf("Run should call two DISTINCT Do methods, got %v", deps)
	}
	if deps[0] == deps[1] {
		t.Errorf("the two Do calls collapsed into one node: %v", deps)
	}
	for _, d := range deps {
		if !strings.Contains(d, "Do") {
			t.Errorf("callee %q is not a Do method", d)
		}
	}
}
