package graph

import (
	"reflect"
	"testing"
)

// Build the fixture graph:  A -> B -> C,  D -> B
// (A calls B, B calls C, D calls B)
func fixture() *Graph {
	g := New()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("D", "B")
	return g
}

// DependsOn / Dependents.
func TestDependsOnAndDependents(t *testing.T) {
	g := fixture()
	if got := g.DependsOn("A"); !reflect.DeepEqual(got, []string{"B"}) {
		t.Errorf("DependsOn(A) = %v, want [B]", got)
	}
	if got := g.Dependents("B"); !reflect.DeepEqual(got, []string{"A", "D"}) {
		t.Errorf("Dependents(B) = %v, want [A D] (sorted)", got)
	}
}

// Impact: blast radius of changing C is everything that reaches C.
func TestImpact(t *testing.T) {
	g := fixture()
	got := g.Impact("C")
	want := []string{"A", "B", "D"} // sorted, excludes C
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Impact(C) = %v, want %v", got, want)
	}
}

// Trace: how does A reach C?
func TestTrace(t *testing.T) {
	g := fixture()
	got := g.Trace("A", "C")
	want := []string{"A", "B", "C"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Trace(A,C) = %v, want %v", got, want)
	}
	if got := g.Trace("C", "A"); got != nil {
		t.Errorf("Trace(C,A) = %v, want nil (unreachable)", got)
	}
}

// Resolve maps bare names onto the qualified ids the typed builder produces —
// without it, `Impact("termScore")` silently returns nothing.
func TestResolveBareNameAgainstQualifiedIDs(t *testing.T) {
	g := New()
	g.AddEdge("(*m/internal/search.BM25).Search", "(*m/internal/search.BM25).termScore")
	g.AddEdge("m/internal/search.NewBM25", "(*m/internal/search.BM25).Search")

	got := g.Resolve("termScore")
	want := "(*m/internal/search.BM25).termScore"
	if len(got) != 1 || got[0] != want {
		t.Fatalf("Resolve(termScore) = %v, want [%s]", got, want)
	}
	// And the resolved id actually has callers, which was the reported bug.
	if callers := g.Impact(got[0]); len(callers) != 2 {
		t.Errorf("Impact(%s) = %v, want both callers", got[0], callers)
	}

	// Exact ids still win outright.
	if got := g.Resolve(want); len(got) != 1 || got[0] != want {
		t.Errorf("Resolve(exact) = %v, want [%s]", got, want)
	}
	// Unknown symbols resolve to nothing.
	if got := g.Resolve("nosuchsymbol"); len(got) != 0 {
		t.Errorf("Resolve(unknown) = %v, want none", got)
	}
}

// Two types sharing a method name are ambiguous: every candidate is returned.
func TestResolveAmbiguousShortName(t *testing.T) {
	g := New()
	g.AddEdge("caller", "(*m/pkg.A).Do")
	g.AddEdge("caller", "(*m/pkg.B).Do")

	got := g.Resolve("Do")
	if len(got) != 2 {
		t.Fatalf("Resolve(Do) = %v, want both candidates", got)
	}
}
