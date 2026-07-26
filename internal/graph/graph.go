// Package graph builds a dependency/call graph and answers impact/trace queries.
//
// Edges are directed from caller to callee (A -> B means "A calls/imports B").
// From that one structure you get:
//   - DependsOn(n): direct callees of n (its out-neighbors)
//   - Dependents(n): direct callers of n (its in-neighbors)
//   - Impact(n): the transitive callers of n (reverse reachability) = the
//     blast radius of changing n
//   - Trace(from,to): a path showing how `from` reaches `to`
//
// Limitation: static edges miss dynamic dispatch (callbacks, interface -> impl,
// dependency injection).
package graph

import (
	"maps"
	"slices"
	"strings"
)

// Graph is a directed graph of string nodes (symbols or files).
type Graph struct {
	out map[string]map[string]bool // node -> set of callees
	in  map[string]map[string]bool // node -> set of callers
}

// New returns an empty graph.
func New() *Graph {
	return &Graph{
		out: map[string]map[string]bool{},
		in:  map[string]map[string]bool{},
	}
}

// AddEdge records "from calls/imports to".
func (g *Graph) AddEdge(from, to string) {
	if g.out[from] == nil {
		g.out[from] = map[string]bool{}
	}
	if g.in[to] == nil {
		g.in[to] = map[string]bool{}
	}
	g.out[from][to] = true
	g.in[to][from] = true
}

// DependsOn returns the direct out-neighbors of n (what n depends on), sorted.
func (g *Graph) DependsOn(n string) []string { return sortedKeys(g.out[n]) }

// Dependents returns the direct in-neighbors of n (who depends on n), sorted.
func (g *Graph) Dependents(n string) []string { return sortedKeys(g.in[n]) }

// Nodes returns every node in the graph (appearing as a source or target), sorted.
func (g *Graph) Nodes() []string {
	set := map[string]bool{}
	for n := range g.out {
		set[n] = true
	}
	for n := range g.in {
		set[n] = true
	}
	return sortedKeys(set)
}

func sortedKeys(set map[string]bool) []string {
	return slices.Sorted(maps.Keys(set))
}

// Resolve maps a symbol name to the node ids it could refer to.
//
// The typed builder uses qualified ids ("(*m/pkg.Type).Method", "m/pkg.Func"),
// so a caller naming a bare "Method" would otherwise match nothing. Resolution
// is progressively looser, stopping at the first tier that matches:
//
//  1. an exact node id;
//  2. nodes whose final identifier equals name — usually one, but two types can
//     share a method name, in which case every candidate is returned;
//  3. nodes containing name, case-insensitively.
//
// Results are sorted (Nodes is), so they're deterministic.
//
// One pass collects all three tiers at once: an exact match returns
// immediately (node ids are unique), a short-name match implies a substring
// match, so the tiers stay disjoint.
func (g *Graph) Resolve(name string) []string {
	if name == "" {
		return nil
	}
	lower := strings.ToLower(name)
	var byShort, bySub []string
	for _, n := range g.Nodes() {
		switch {
		case n == name:
			return []string{n}
		case shortName(n) == name:
			byShort = append(byShort, n)
		case strings.Contains(strings.ToLower(n), lower):
			bySub = append(bySub, n)
		}
	}
	if len(byShort) > 0 {
		return byShort
	}
	return bySub
}

// shortName is the final identifier of a node id:
//
//	"(*m/pkg.BM25).termScore" -> "termScore"
//	"m/pkg.Func"              -> "Func"
//	"termScore"               -> "termScore"
func shortName(id string) string {
	if i := strings.LastIndex(id, "."); i >= 0 {
		return id[i+1:]
	}
	return id
}

// Impact returns every node that can transitively reach n by following edges
// backward — the full set of callers affected if n changes. Excludes n itself.
func (g *Graph) Impact(n string) []string {
	seen := map[string]bool{n: true} // pre-seed n so it's skipped, and cycles terminate
	queue := []string{n}
	var callers []string
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for caller := range g.in[cur] { // BACKWARD: who points at cur
			if seen[caller] {
				continue
			}
			seen[caller] = true
			callers = append(callers, caller)
			queue = append(queue, caller)
		}
	}
	slices.Sort(callers)
	return callers
}

// Trace returns a path (node sequence) from `from` to `to` following edges
// forward, or nil if `to` is unreachable from `from`.
func (g *Graph) Trace(from, to string) []string {
	if from == to {
		return []string{from}
	}
	queue := []string{from}
	visited := map[string]bool{from: true}
	prev := map[string]string{} // node -> the node we discovered it from
	found := false
	for len(queue) > 0 && !found {
		n := queue[0]
		queue = queue[1:]                   // pop FRONT (FIFO)
		for _, nb := range g.DependsOn(n) { // sorted -> deterministic shortest path
			if visited[nb] {
				continue
			}
			visited[nb] = true
			prev[nb] = n
			if nb == to {
				found = true
				break
			}
			queue = append(queue, nb)
		}
	}
	if !found {
		return nil // `to` is unreachable from `from`
	}
	// Walk predecessors from `to` back to `from`, then reverse into forward order.
	path := []string{to}
	for cur := to; cur != from; {
		cur = prev[cur]
		path = append(path, cur)
	}
	slices.Reverse(path)
	return path
}
