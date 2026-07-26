package store

import (
	"container/heap"
	"sort"

	"github.com/jKiler/descry/internal/core"
	"github.com/jKiler/descry/internal/embed"
)

// MemStore is the in-memory implementation. It keeps the chunks (with their
// float32 vectors) and a quantized int8 vector matrix; queries scan the int8
// matrix and rerank the top candidates with exact float32 cosine.
type MemStore struct {
	chunks []core.Chunk

	// rerankMult overrides defaultRerankMult when > 0.
	rerankMult int

	// Quantized vector index, (re)built lazily to match chunks. q8 is a flat
	// row-major [len(chunks) × dim] int8 matrix; an unembedded chunk's row is
	// left zero and skipped at query time. q8rows records how many chunks q8
	// reflects, so a stale index (after Add/AddBatch) is rebuilt on next query.
	q8     []int8
	dim    int
	q8rows int
}

// NewMemStore returns an empty in-memory store.
func NewMemStore() *MemStore { return &MemStore{} }

// SetRerankMult sets how many candidates per result the int8 pre-filter keeps
// for exact reranking (see defaultRerankMult). Values <= 0 restore the default.
func (s *MemStore) SetRerankMult(m int) { s.rerankMult = m }

// Add appends a chunk.
func (s *MemStore) Add(c core.Chunk) error {
	s.chunks = append(s.chunks, c)
	return nil
}

// AddBatch appends many chunks (no transaction needed in memory).
func (s *MemStore) AddBatch(chunks []core.Chunk) error {
	s.chunks = append(s.chunks, chunks...)
	return nil
}

// All returns every stored chunk.
func (s *MemStore) All() []core.Chunk { return s.chunks }

// Len returns the number of stored chunks.
func (s *MemStore) Len() int { return len(s.chunks) }

// Nearest returns the topK chunks whose vectors are most cosine-similar to the
// query, highest first. It runs in two stages: a fast int8 dot-product scan over
// the quantized matrix selects a candidate pool (topK * rerankMult), then those
// candidates are reranked with exact float32 cosine. The result is exact for
// everything that clears the deliberately generous pre-filter, at a fraction of
// a full float scan. Falls back to an exact scan when there is no usable
// quantized index (no embedded chunks, or a query of a different dimension).
func (s *MemStore) Nearest(vector []float32, topK int) []core.SearchResult {
	if topK <= 0 {
		return nil
	}
	s.ensureQuantized()
	if s.dim == 0 || len(vector) != s.dim {
		return s.nearestExact(vector, topK)
	}

	query := quantize(vector)

	mult := s.rerankMult
	if mult <= 0 {
		mult = defaultRerankMult
	}
	pool := topK * mult

	// Stage 1: int8 dot scan, keeping only the pool best in a bounded min-heap
	// (the root is the worst kept, so most chunks cost one comparison) instead
	// of scoring, sorting, and discarding the whole corpus.
	cands := make(candHeap, 0, pool)
	for i := range s.chunks {
		if len(s.chunks[i].Vector) != s.dim {
			continue // unembedded — its q8 row is zero
		}
		c := cand{doc: i, approx: dotI8(s.q8[i*s.dim:(i+1)*s.dim], query)}
		switch {
		case len(cands) < pool:
			heap.Push(&cands, c)
		case c.approx > cands[0].approx:
			cands[0] = c
			heap.Fix(&cands, 0)
		}
	}

	// Stage 2: exact float32 rerank of the pool (order doesn't matter here —
	// topSorted sorts by the exact score).
	scored := make([]core.SearchResult, len(cands))
	for i, c := range cands {
		scored[i] = core.SearchResult{
			Chunk: s.chunks[c.doc],
			Score: embed.Cosine(vector, s.chunks[c.doc].Vector),
		}
	}
	return topSorted(scored, topK)
}

// cand is one stage-1 candidate: a chunk index and its approximate (int8 dot)
// score.
type cand struct {
	doc    int
	approx int32
}

// candHeap is a min-heap of candidates by approximate score, used to keep the
// best pool during the stage-1 scan.
type candHeap []cand

func (h candHeap) Len() int           { return len(h) }
func (h candHeap) Less(i, j int) bool { return h[i].approx < h[j].approx }
func (h candHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *candHeap) Push(x any)        { *h = append(*h, x.(cand)) }
func (h *candHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// nearestExact is the brute-force float32 fallback — also the reference the
// quantized path is tested against.
func (s *MemStore) nearestExact(vector []float32, topK int) []core.SearchResult {
	scored := make([]core.SearchResult, 0, len(s.chunks))
	for _, chunk := range s.chunks {
		if len(chunk.Vector) == 0 {
			continue // unembedded chunk — skip
		}
		scored = append(scored, core.SearchResult{Chunk: chunk, Score: embed.Cosine(vector, chunk.Vector)})
	}
	return topSorted(scored, topK)
}

// topSorted sorts by score descending (ties broken by chunk ID) and returns a
// right-sized copy of the first topK, so the full backing array can be freed.
func topSorted(scored []core.SearchResult, topK int) []core.SearchResult {
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].Score != scored[j].Score {
			return scored[i].Score > scored[j].Score
		}
		return scored[i].Chunk.ID < scored[j].Chunk.ID
	})
	topK = min(topK, len(scored))
	out := make([]core.SearchResult, topK)
	copy(out, scored)
	return out
}

// ensureQuantized (re)builds the int8 matrix when it no longer matches the
// current chunks. dim comes from the first embedded chunk; unembedded chunks get
// a zero row (skipped at query time).
func (s *MemStore) ensureQuantized() {
	if s.q8rows == len(s.chunks) {
		return
	}
	dim := 0
	for i := range s.chunks {
		if n := len(s.chunks[i].Vector); n > 0 {
			dim = n
			break
		}
	}
	s.dim = dim
	s.q8rows = len(s.chunks)
	if dim == 0 {
		s.q8 = nil
		return
	}
	s.q8 = make([]int8, len(s.chunks)*dim)
	for i := range s.chunks {
		if v := s.chunks[i].Vector; len(v) == dim {
			quantizeInto(s.q8[i*dim:(i+1)*dim], v)
		}
	}
}

var _ Store = (*MemStore)(nil)
