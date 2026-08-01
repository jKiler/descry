package chunk

import (
	"fmt"
	"sort"
)

// Default returns the chunker the pipeline uses unless told otherwise. Callers
// that just want "the good one" should use this rather than naming a strategy,
// so swapping the default is a one-line change here.
//
// The tree-sitter chunker earned this on measured numbers, including on Go —
// the one language that already had a real AST chunker. Against `ast-go` on the
// kubernetes corpus it takes chunk Recall@20 from 22.7% to 32.0% with file
// recall unchanged, because `go/ast` emits one chunk per declaration with no
// size limit and a long function's tail never reaches the embedder. Six of the
// seven corpora improve.
func Default() Chunker { return NewTSChunker() }

// builders maps a strategy name to its constructor. The harness names a
// strategy on the command line to A/B two chunkers over the same corpora, and
// the name is also the fingerprint component that keeps their indexes apart.
var builders = map[string]func() Chunker{
	"line":   func() Chunker { return NewLineChunker() },
	"ast-go": func() Chunker { return NewASTChunker() },
}

// register adds a chunker strategy. Called from init in the file that defines
// the strategy, so a new chunker is self-registering.
func register(name string, build func() Chunker) {
	if _, dup := builders[name]; dup {
		panic("chunk: duplicate chunker name " + name)
	}
	builders[name] = build
}

// ByName returns the named chunker, or the default when name is empty.
func ByName(name string) (Chunker, error) {
	if name == "" {
		return Default(), nil
	}
	b, ok := builders[name]
	if !ok {
		return nil, fmt.Errorf("unknown chunker %q (have: %v)", name, Names())
	}
	return b(), nil
}

// Names lists the registered chunker strategies.
func Names() []string {
	out := make([]string, 0, len(builders))
	for n := range builders {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}
