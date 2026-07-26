package graph

import (
	"cmp"
	"fmt"
	"slices"
	"strings"
)

// ToMermaid renders the graph as a Mermaid flowchart. Paste the output into any
// Mermaid renderer (mermaid.live, GitHub markdown, VS Code preview) to see the
// call/dependency graph.
//
// Each node gets a synthetic id (n0, n1, …) with the real name as a quoted
// label, so arbitrary names render safely — including the typed builder's
// qualified ids like "(m/pkg.Type).Method", which contain dots and parens that
// aren't valid as bare Mermaid ids. Edges are sorted for deterministic output.
func (g *Graph) ToMermaid() string {
	nodes := g.Nodes()
	id := make(map[string]string, len(nodes))
	for i, n := range nodes {
		id[n] = fmt.Sprintf("n%d", i)
	}

	type edge struct{ from, to string }
	var edges []edge
	for from, tos := range g.out {
		for to := range tos {
			edges = append(edges, edge{from, to})
		}
	}
	slices.SortFunc(edges, func(a, b edge) int {
		if a.from != b.from {
			return cmp.Compare(a.from, b.from)
		}
		return cmp.Compare(a.to, b.to)
	})

	var b strings.Builder
	b.WriteString("flowchart LR\n")
	for _, e := range edges {
		// %q quotes the label; Mermaid's id["label"] form accepts dots/parens/slashes.
		fmt.Fprintf(&b, "  %s[%q] --> %s[%q]\n", id[e.from], e.from, id[e.to], e.to)
	}
	return b.String()
}
