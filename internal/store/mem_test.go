package store

import (
	"testing"

	"github.com/jKiler/descry/internal/core"
)

// MemStore nearest-neighbor behavior. Run: go test ./internal/store
func TestMemStore_NearestRanksByCosine(t *testing.T) {
	s := NewMemStore()
	// Simple 3-d vectors; the query points along the "north" axis.
	s.Add(core.Chunk{ID: "north", Vector: []float32{1, 0, 0}})
	s.Add(core.Chunk{ID: "tilt", Vector: []float32{0.8, 0.6, 0}})
	s.Add(core.Chunk{ID: "east", Vector: []float32{0, 1, 0}})
	s.Add(core.Chunk{ID: "novec"}) // nil vector: must be skipped, not panic

	got := s.Nearest([]float32{1, 0, 0}, 2)
	if len(got) != 2 {
		t.Fatalf("want 2 results, got %d", len(got))
	}
	if got[0].Chunk.ID != "north" {
		t.Errorf("top result = %q, want north", got[0].Chunk.ID)
	}
	if got[1].Chunk.ID != "tilt" {
		t.Errorf("second result = %q, want tilt", got[1].Chunk.ID)
	}
	if !(got[0].Score >= got[1].Score) {
		t.Errorf("results not sorted descending: %.3f then %.3f", got[0].Score, got[1].Score)
	}
}

// Regression: topK larger than the number of embedded chunks must clamp (no
// slice-bounds panic), and unembedded chunks must never appear in results.
func TestMemStore_NearestClampsAndSkipsUnembedded(t *testing.T) {
	s := NewMemStore()
	s.Add(core.Chunk{ID: "a", Vector: []float32{1, 0}})
	s.Add(core.Chunk{ID: "b", Vector: []float32{0, 1}})
	s.Add(core.Chunk{ID: "novec"}) // nil vector

	got := s.Nearest([]float32{1, 0}, 10) // topK > embedded count
	if len(got) != 2 {
		t.Fatalf("want 2 results (novec skipped, topK clamped), got %d: %+v", len(got), got)
	}
	for _, r := range got {
		if r.Chunk.ID == "novec" {
			t.Errorf("unembedded chunk must not appear in results")
		}
	}
}

// Empty store must not panic and returns no results.
func TestMemStore_NearestEmpty(t *testing.T) {
	if got := NewMemStore().Nearest([]float32{1, 0}, 5); len(got) != 0 {
		t.Errorf("empty store should return 0 results, got %d", len(got))
	}
}

// The quantized int8 scan + float32 rerank must return the same top-K as an
// exact float32 scan — that's the whole point of the two-stage design. Checked
// on a random normalized corpus larger than the rerank pool, so the int8
// pre-filter genuinely has to select the right candidates.
func TestMemStore_QuantizedMatchesExactTopK(t *testing.T) {
	const (
		n    = 800
		dim  = 64
		topK = 20
	)
	rng := newRand(7)
	s := NewMemStore()
	for i := 0; i < n; i++ {
		s.chunks = append(s.chunks, core.Chunk{ID: padID(i), Vector: randUnit(rng, dim)})
	}

	for q := 0; q < 25; q++ {
		query := randUnit(rng, dim)
		want := s.nearestExact(query, topK)
		got := s.Nearest(query, topK)
		if len(got) != len(want) {
			t.Fatalf("query %d: got %d results, want %d", q, len(got), len(want))
		}
		for i := range want {
			if got[i].Chunk.ID != want[i].Chunk.ID {
				t.Errorf("query %d rank %d: quantized=%s exact=%s (scores %.5f vs %.5f)",
					q, i, got[i].Chunk.ID, want[i].Chunk.ID, got[i].Score, want[i].Score)
			}
		}
	}
}

// A larger rerank pool never makes results worse; a pool of 1 still returns the
// single best result (rerank of the top-1 int8 candidate).
func TestMemStore_RerankMultBounds(t *testing.T) {
	rng := newRand(3)
	s := NewMemStore()
	for i := 0; i < 200; i++ {
		s.chunks = append(s.chunks, core.Chunk{ID: padID(i), Vector: randUnit(rng, 32)})
	}
	query := randUnit(rng, 32)
	best := s.nearestExact(query, 1)

	s.SetRerankMult(1)
	got := s.Nearest(query, 1)
	if len(got) != 1 || got[0].Chunk.ID != best[0].Chunk.ID {
		t.Errorf("rerankMult=1 top-1 = %v, want %s", got, best[0].Chunk.ID)
	}
}

// --- test helpers ---

func newRand(seed uint64) func() float64 {
	s := seed
	return func() float64 {
		s ^= s << 13
		s ^= s >> 7
		s ^= s << 17
		return float64(s>>11) / float64(1<<53)
	}
}

func randUnit(rng func() float64, dim int) []float32 {
	v := make([]float32, dim)
	var sum float64
	for i := range v {
		x := rng()*2 - 1
		v[i] = float32(x)
		sum += x * x
	}
	norm := float32(1)
	if sum > 0 {
		norm = float32(1 / (sqrt(sum)))
	}
	for i := range v {
		v[i] *= norm
	}
	return v
}

func sqrt(x float64) float64 {
	// avoid importing math in the test for one call
	z := x
	for i := 0; i < 40; i++ {
		if z == 0 {
			break
		}
		z = (z + x/z) / 2
	}
	return z
}

func padID(i int) string {
	const w = 5
	b := []byte("00000")
	for p := w - 1; p >= 0 && i > 0; p-- {
		b[p] = byte('0' + i%10)
		i /= 10
	}
	return string(b)
}
