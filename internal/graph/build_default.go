package graph

// Build constructs the call graph with the preferred resolver: it tries the
// type-checked builder (BuildFromGoDirTyped) first and falls back to the
// name-based heuristic (BuildFromGoDir) when the typed build fails or yields no
// nodes — e.g. root isn't a module, or nothing type-checks. The returned typed
// flag reports which resolver produced the graph. An error is returned only if
// the fallback also fails.
func Build(root string) (g *Graph, typed bool, err error) {
	if tg, terr := BuildFromGoDirTyped(root); terr == nil && len(tg.Nodes()) > 0 {
		return tg, true, nil
	}
	ng, nerr := BuildFromGoDir(root)
	return ng, false, nerr
}
