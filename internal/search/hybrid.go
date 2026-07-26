package search

import (
	"github.com/jKiler/descry/internal/core"
	"github.com/jKiler/descry/internal/embed"
	"github.com/jKiler/descry/internal/store"
)

// Hybrid is the full retriever: embed the query, get a vector ranking from the
// store and a lexical ranking from BM25, then fuse them with RRF. This is what
// the CLI's search command calls.
//
// Construct with NewHybrid — the zero value carries no usable defaults.
type Hybrid struct {
	Emb   embed.Embedder
	Store store.Store
	Lex   *BM25
	RRFK  float64 // the RRF constant (NewHybrid sets 60)

	// Per-list RRF weights. Vectors carry slightly more than full weight (the
	// strongest single ranker, and max-leaning fusion rewards that); the
	// chunk-level BM25 is down-weighted because the whole-file BM25 covers most
	// of what it did. The optimum is workload-dependent; all are overridable at
	// runtime; README "Measured quality" records the sweep results.
	VecWeight float64
	LexWeight float64

	// CandMult controls how deep each ranker's candidate list goes relative to
	// topK (each list retrieves topK*CandMult before fusion). Deeper lists give
	// RRF more chances to surface a doc that one ranker likes and the other
	// missed. NewHybrid sets 2.
	CandMult int

	// LexFile, if set, is a BM25 over whole-file documents (see FileDocs) that
	// SearchFiles fuses as a third ranking beside the vector and chunk-level
	// lexical lists. The two lexical views are complementary: whole-file BM25
	// measures tf and length on the real unit of retrieval, while the chunk
	// view still surfaces a single sharply-matching declaration inside a large
	// file that whole-file length normalization would dilute.
	LexFile *BM25

	// LexFileWeight is LexFile's RRF weight in SearchFiles (the chunk-level
	// lexical list keeps LexWeight). NewHybrid sets 1.0.
	LexFileWeight float64

	// FuseAlpha is the consensus dial for SearchFiles' fusion (see
	// ReciprocalRankFusionAlpha): 1 = plain RRF sum, 0 = best list only.
	// NewHybrid sets 1.0.
	FuseAlpha float64
}

// NewHybrid builds a hybrid retriever over an already-populated store and BM25.
// The defaults are the kubernetes-120 sweep optimum (q8: R@10 95.8%, MRR 0.720;
// fp32 within noise of its own optimum — README "Measured quality"); set LexFile to enable
// the whole-file lexical list.
func NewHybrid(e embed.Embedder, s store.Store, lex *BM25) *Hybrid {
	return &Hybrid{Emb: e, Store: s, Lex: lex,
		RRFK: 25, VecWeight: 1.1, LexWeight: 0.5, LexFileWeight: 1.0, FuseAlpha: 0.40, CandMult: 8}
}

// Search runs vector + lexical retrieval and fuses the two rankings.
//
// Search embeds the query, gets a vector ranking from the store and a lexical
// ranking from BM25, fuses them with RRF, and returns the top topK — with the
// real chunks reattached so callers get content and line ranges.
func (h *Hybrid) Search(query string, topK int) []core.SearchResult {
	qv := h.Emb.Embed(query)
	vecRes := h.Store.Nearest(qv, topK*h.CandMult)
	lexRes := h.Lex.Search(query, topK*h.CandMult)

	// Extract ranked ID lists, remembering each id's chunk for reconstruction.
	byID := make(map[string]core.Chunk)
	rankedIDs := func(rs []core.SearchResult) []string {
		out := make([]string, len(rs))
		for i, r := range rs {
			out[i] = r.Chunk.ID
			byID[r.Chunk.ID] = r.Chunk
		}
		return out
	}
	vecIDs := rankedIDs(vecRes)
	lexIDs := rankedIDs(lexRes)

	fused := ReciprocalRankFusion([][]string{vecIDs, lexIDs}, h.RRFK,
		[]float64{h.VecWeight, h.LexWeight})

	topK = min(max(topK, 0), len(fused))
	results := make([]core.SearchResult, topK)
	for i := range topK {
		results[i] = core.SearchResult{
			Chunk: byID[fused[i].ID],
			Score: fused[i].Score, // real fused RRF confidence, not a rank placeholder
		}
	}
	return results
}

// SearchFiles is Search at file granularity: each ranker's chunk list is
// collapsed to files BEFORE fusion (a file's rank = its best chunk's rank), and
// RRF then fuses the two file rankings. Fusing after the collapse matters:
// vector search and BM25 often like *different* chunks of the same file, and
// chunk-level fusion splits that file's votes across chunk ids, crediting
// neither. One result per file is returned, best first, carrying the file's
// best chunk (preferring the vector ranking's pick for content).
func (h *Hybrid) SearchFiles(query string, topK int) []core.SearchResult {
	qv := h.Emb.Embed(query)
	vecRes := h.Store.Nearest(qv, topK*h.CandMult)
	lexRes := h.Lex.Search(query, topK*h.CandMult)

	// Collapse a chunk ranking to a file ranking: the first occurrence of a path
	// is that file's best chunk, later occurrences are dropped. Each list keeps
	// its own ranking (a file in both must appear in both, or fusion can't
	// credit the agreement); byPath just remembers a representative chunk.
	byPath := make(map[string]core.Chunk)
	collapse := func(rs []core.SearchResult) []string {
		seen := make(map[string]bool)
		var paths []string
		for _, r := range rs {
			if seen[r.Chunk.Path] {
				continue
			}
			seen[r.Chunk.Path] = true
			if _, ok := byPath[r.Chunk.Path]; !ok {
				byPath[r.Chunk.Path] = r.Chunk
			}
			paths = append(paths, r.Chunk.Path)
		}
		return paths
	}
	vecPaths := collapse(vecRes) // fills byPath first, so vector picks win
	lexPaths := collapse(lexRes)

	lists := [][]string{vecPaths, lexPaths}
	weights := []float64{h.VecWeight, h.LexWeight}
	if h.LexFile != nil {
		lists = append(lists, collapse(h.LexFile.Search(query, topK*h.CandMult)))
		weights = append(weights, h.LexFileWeight)
	}
	fused := ReciprocalRankFusionAlpha(lists, h.RRFK, weights, h.FuseAlpha)

	topK = min(max(topK, 0), len(fused))
	results := make([]core.SearchResult, topK)
	for i := range topK {
		results[i] = core.SearchResult{Chunk: byPath[fused[i].ID], Score: fused[i].Score}
	}
	return results
}
