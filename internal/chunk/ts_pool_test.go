package chunk

import (
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/jKiler/descry/internal/lang"
)

// A ts.Parser owns C memory that only Close frees, and the binding installs no
// finalizer — so every parser this pool creates must end up either back in the
// pool or explicitly closed. A sync.Pool could not promise that: it drops its
// contents on any GC, and a dropped parser leaks the C allocation behind it.
//
// The pool is bounded by GOMAXPROCS, so after any amount of concurrent parsing
// the number of live parsers is bounded too. This checks the invariant the
// channel gives us and a sync.Pool did not.
func TestParserPoolIsBoundedAndReuses(t *testing.T) {
	p := lang.ByName("go")
	if p == nil || p.Grammar == nil {
		t.Skip("go pack unavailable")
	}
	pools := newParserPools()
	src := []byte("package a\n\nfunc f() { return }\n")

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				if tree := pools.parse(p, p.Grammar(), src); tree != nil {
					tree.Close()
				}
			}
		}()
	}
	wg.Wait()

	pool := pools.poolFor(p)
	if cap(pool) != runtime.GOMAXPROCS(0) {
		t.Errorf("pool capacity %d, want GOMAXPROCS %d", cap(pool), runtime.GOMAXPROCS(0))
	}
	if len(pool) == 0 {
		t.Error("pool is empty after 400 parses; parsers are not being returned for reuse")
	}
	if len(pool) > cap(pool) {
		t.Errorf("pool holds %d parsers, over its capacity %d", len(pool), cap(pool))
	}
}

// A pack whose grammar cannot be set must not leave a parser behind, and must
// degrade rather than panic.
func TestParserPoolHandlesUnusableGrammar(t *testing.T) {
	p := &lang.Pack{Name: "broken-for-test"}
	pools := newParserPools()
	if tree := pools.parse(p, nil, []byte("x")); tree != nil {
		tree.Close()
		t.Error("a nil language produced a tree")
	}
	if len(pools.poolFor(p)) != 0 {
		t.Error("a parser whose language could not be set was pooled anyway")
	}
}

// The chunker must survive a file it cannot parse, from many goroutines at
// once, without losing content: every file still yields chunks via the fallback.
func TestChunkerConcurrentOnUnparsableInput(t *testing.T) {
	c := NewTSChunker()
	garbage := strings.Repeat("func ((( {{{ \x00 unterminated\n", 40)

	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for range 20 {
				if got := c.Chunk("a.go", garbage); len(got) == 0 {
					t.Errorf("worker %d: unparsable file produced no chunks", i)
					return
				}
			}
		}(i)
	}
	wg.Wait()
}
