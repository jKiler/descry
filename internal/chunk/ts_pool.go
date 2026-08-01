package chunk

import (
	"runtime"
	"sync"

	"github.com/jKiler/descry/internal/lang"
	ts "github.com/tree-sitter/go-tree-sitter"
)

// Parsers and compiled queries are expensive to build — a parser is a few
// milliseconds and a query is a grammar-sized compile — and indexing calls the
// chunker once per file from GOMAXPROCS workers. Both are therefore cached: the
// query once per pack for the process's life, the parser in a per-pack pool so
// each worker keeps its own hot instance without a lock on the hot path.

var (
	queryMu    sync.Mutex
	queryCache = map[*lang.Pack]*compiledQuery{}
	queryErrs  = map[*lang.Pack]error{}
)

// compiledFor returns the pack's compiled query, compiling it on first use. A
// query that fails to compile is remembered as failed: a broken pack degrades
// its language to the fallback chunker instead of re-attempting the compile for
// every file in the repository.
func compiledFor(p *lang.Pack) (*compiledQuery, error) {
	queryMu.Lock()
	defer queryMu.Unlock()
	if q, ok := queryCache[p]; ok {
		return q, nil
	}
	if err, ok := queryErrs[p]; ok {
		return nil, err
	}
	q, err := newCompiled(p)
	if err != nil {
		queryErrs[p] = err
		return nil, err
	}
	queryCache[p] = q
	return q, nil
}

// parserPools holds one bounded pool of parsers per language.
//
// Deliberately a channel and not a sync.Pool. A ts.Parser owns memory allocated
// by C (ts_parser_new), the binding installs no finalizer, and sync.Pool
// discards its contents on every GC cycle — so a pooled parser that the GC drops
// frees the Go struct and leaks the C parser behind it, once per dropped
// parser for as long as indexing runs. A channel never drops anything: a parser
// is either in the pool, in use, or explicitly Closed.
type parserPools struct {
	mu    sync.Mutex
	pools map[*lang.Pack]chan *ts.Parser
}

func newParserPools() *parserPools {
	return &parserPools{pools: map[*lang.Pack]chan *ts.Parser{}}
}

// poolFor returns the pack's parser pool, sized to the indexer's worker count —
// beyond that, parsers would sit idle rather than be reused.
func (pp *parserPools) poolFor(p *lang.Pack) chan *ts.Parser {
	pp.mu.Lock()
	defer pp.mu.Unlock()
	pool, ok := pp.pools[p]
	if !ok {
		pool = make(chan *ts.Parser, runtime.GOMAXPROCS(0))
		pp.pools[p] = pool
	}
	return pool
}

// parse parses src with a pooled parser, returning nil if the language could
// not be set or the parse produced no tree. The caller owns Close on the tree.
func (pp *parserPools) parse(p *lang.Pack, l *ts.Language, src []byte) *ts.Tree {
	if l == nil {
		return nil // SetLanguage dereferences it; a caller's bad pack must not panic here
	}
	pool := pp.poolFor(p)

	var parser *ts.Parser
	select {
	case parser = <-pool:
	default:
		parser = ts.NewParser()
		if err := parser.SetLanguage(l); err != nil {
			parser.Close()
			return nil
		}
	}

	tree := parser.Parse(src, nil)
	// Reset before returning the parser: a parser holds a reference to the text
	// it last parsed, and pooling it without reset would pin that buffer.
	parser.Reset()
	select {
	case pool <- parser:
	default:
		parser.Close() // more parsers live than the pool holds; free this one
	}
	return tree
}

var (
	_ Chunker    = (*TSChunker)(nil)
	_ Diagnostic = (*TSChunker)(nil)
)
