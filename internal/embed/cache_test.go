package embed

import (
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

// countingEmbedder wraps HashEmbedder and counts real Embed calls, so tests can
// prove whether the cache actually short-circuited the model.
type countingEmbedder struct {
	inner Embedder
	calls atomic.Int64
}

func (c *countingEmbedder) Dim() int   { return c.inner.Dim() }
func (c *countingEmbedder) ID() string { return c.inner.ID() }
func (c *countingEmbedder) Embed(text string) []float32 {
	c.calls.Add(1)
	return c.inner.Embed(text)
}

func TestCachedEmbedderHitsSkipInner(t *testing.T) {
	inner := &countingEmbedder{inner: NewHashEmbedder(64)}
	ce, err := NewCachedEmbedder(inner, filepath.Join(t.TempDir(), "cache.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer ce.Close()

	first := ce.Embed("some chunk text")
	second := ce.Embed("some chunk text")
	if got := inner.calls.Load(); got != 1 {
		t.Errorf("inner embedder called %d times, want 1 (second call should hit cache)", got)
	}
	if len(first) != len(second) {
		t.Fatalf("vector lengths differ: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("cached vector differs at %d: %v vs %v", i, first[i], second[i])
		}
	}
}

func TestCachedEmbedderPersistsAcrossReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.db")

	inner1 := &countingEmbedder{inner: NewHashEmbedder(64)}
	ce1, err := NewCachedEmbedder(inner1, path)
	if err != nil {
		t.Fatal(err)
	}
	want := ce1.Embed("persisted text")
	ce1.Close()

	inner2 := &countingEmbedder{inner: NewHashEmbedder(64)}
	ce2, err := NewCachedEmbedder(inner2, path)
	if err != nil {
		t.Fatal(err)
	}
	defer ce2.Close()
	got := ce2.Embed("persisted text")
	if inner2.calls.Load() != 0 {
		t.Errorf("inner embedder called after reopen — cache did not persist")
	}
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("persisted vector differs at %d", i)
		}
	}
}

func TestCachedEmbedderTransparentIdentity(t *testing.T) {
	inner := NewHashEmbedder(64)
	ce, err := NewCachedEmbedder(inner, filepath.Join(t.TempDir(), "cache.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer ce.Close()
	if ce.ID() != inner.ID() || ce.Dim() != inner.Dim() {
		t.Errorf("cache must be identity-transparent: got %q/%d, want %q/%d",
			ce.ID(), ce.Dim(), inner.ID(), inner.Dim())
	}
}

func TestCachedEmbedderConcurrent(t *testing.T) {
	inner := &countingEmbedder{inner: NewHashEmbedder(64)}
	ce, err := NewCachedEmbedder(inner, filepath.Join(t.TempDir(), "cache.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer ce.Close()

	// The index worker pool calls Embed concurrently; run with -race.
	texts := []string{"alpha", "beta", "gamma", "delta"}
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				if v := ce.Embed(texts[i%len(texts)]); len(v) != ce.Dim() {
					t.Errorf("bad vector length %d", len(v))
					return
				}
			}
		}()
	}
	wg.Wait()
}
