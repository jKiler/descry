package search

import (
	"math"

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
	RRFK  float64 // the RRF constant (NewHybrid sets 25)

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
	// missed. NewHybrid sets 8.
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
	// NewHybrid sets 0.40.
	FuseAlpha float64

	// Rerank, when set, chooses which chunks represent each returned file. It
	// never touches the file ranking — see rerankSelections, which every
	// retrieval path reaches through SearchFilesN.
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
// RRF then fuses the file rankings. Fusing after the collapse matters: vector
// search and BM25 often like *different* chunks of the same file, and
// chunk-level fusion splits that file's votes across chunk ids, crediting
// neither. One result per file is returned, best first, carrying that file's
// best-scoring chunk.
//
// Which chunk a file carries is a second ranking problem, and answering it with
// the same evidence as the first is what the harness measures. It used to be
// answered by whichever list mentioned the path first — in practice the vector
// list, since it was collapsed first — so a file could rank first on the
// agreement of the two lexical rankers and still be represented by the vector
// ranking's unrelated pick. Measured over the 523-query benchmark, 269 queries
// (51.4%) ranked the right file in the top 10 while holding an indexed chunk
// that covered the answer, and showed a different one.
//
// It is the one-chunk projection of SearchFilesN rather than a second
// implementation of it. The two used to duplicate their retrieval and fusion,
// with a doc comment asserting the file rankings were identical and a unit test
// checking that on a fixture too small to reach the whole-file lexical branch —
// the one place the two bodies differed most. Deriving one from the other makes
// the identity structural, which is what it always claimed to be.
func (h *Hybrid) SearchFiles(query string, topK int) []core.SearchResult {
	hits := h.SearchFilesN(query, topK, 1)
	results := make([]core.SearchResult, 0, len(hits))
	for _, fh := range hits {
		if len(fh.Chunks) == 0 {
			continue // no ranker offered a chunk for this path; nothing to show
		}
		results = append(results, core.SearchResult{Chunk: fh.Chunks[0], Score: fh.Score})
	}
	return results
}

// FileHit is one file in a SearchFilesN result: its fused rank score and the
// best n chunks of it, best first.
type FileHit struct {
	Path   string
	Score  float64
	Chunks []core.Chunk
}

// SearchFilesN is SearchFiles with each file carrying up to n of its
// best-scoring chunks instead of exactly one.
//
// It exists because "the right file, the wrong part of it" is this project's
// measured failure mode, and every attempt to fix it by choosing a better
// single chunk has failed — three of them, the last being a cross-encoder
// measured. Widening the result is the other way to answer the same
// complaint, and it needs no better ranker: it needs the caller to accept more
// text.
//
// SearchFiles is this function at n = 1, so the two cannot disagree about the
// file ranking or about which single chunk a file shows;
// TestSearchFilesNAtOneMatchesSearchFiles keeps that visible as a test rather
// than only as a call graph.
//
// The trade this exposes is context, not compute: no extra retrieval or model
// call happens, but a caller showing three chunks per file pays roughly three
// times the tokens for whatever the second and third add. That curve is what
// the harness measures; it is not free, and it should not be presented as free.
func (h *Hybrid) SearchFilesN(query string, topK, n int) []FileHit {
	n = max(n, 1)
	qv := h.Emb.Embed(query)
	vecRes := h.Store.Nearest(qv, topK*h.CandMult)
	lexRes := h.Lex.Search(query, topK*h.CandMult)

	byPath := h.chunksPerPath(n, vecRes, lexRes)
	lists := [][]string{collapsePaths(vecRes), collapsePaths(lexRes)}
	weights := []float64{h.VecWeight, h.LexWeight}
	if h.LexFile != nil {
		fileRes := h.LexFile.Search(query, topK*h.CandMult)
		lists = append(lists, collapsePaths(fileRes))
		weights = append(weights, h.LexFileWeight)
		for _, r := range fileRes {
			if len(byPath[r.Chunk.Path]) == 0 {
				byPath[r.Chunk.Path] = []core.Chunk{r.Chunk}
			}
		}
	}
	fused := ReciprocalRankFusionAlpha(lists, h.RRFK, weights, h.FuseAlpha)

	topK = min(max(topK, 0), len(fused))
	out := make([]FileHit, topK)
	for i := range topK {
		out[i] = FileHit{Path: fused[i].ID, Score: fused[i].Score, Chunks: byPath[fused[i].ID]}
	}
	return out
}

// chunksPerPath fuses the chunk rankings and gives each path its n
// highest-scoring chunks, best first. It is bestChunkPerPath generalized: at
// n = 1 it keeps exactly the chunk that one would have chosen, because it walks
// the same fused list in the same order.
func (h *Hybrid) chunksPerPath(n int, lists ...[]core.SearchResult) map[string][]core.Chunk {
	byID := make(map[string]core.Chunk)
	ranked := make([][]string, len(lists))
	for i, rs := range lists {
		ids := make([]string, len(rs))
		for j, r := range rs {
			ids[j] = r.Chunk.ID
			byID[r.Chunk.ID] = r.Chunk
		}
		ranked[i] = ids
	}

	byPath := make(map[string][]core.Chunk)
	for _, f := range ReciprocalRankFusionAlpha(ranked, h.RRFK,
		[]float64{h.VecWeight, h.LexWeight}, h.FuseAlpha) {
		c := byID[f.ID]
		if len(byPath[c.Path]) < n {
			byPath[c.Path] = append(byPath[c.Path], c)
		}
	}
	return byPath
}

// collapsePaths turns a chunk ranking into a file ranking: first occurrence of
// a path wins, later ones are dropped.
func collapsePaths(rs []core.SearchResult) []string {
	seen := make(map[string]bool)
	var paths []string
	for _, r := range rs {
		if seen[r.Chunk.Path] {
			continue
		}
		seen[r.Chunk.Path] = true
		paths = append(paths, r.Chunk.Path)
	}
	return paths
}

// allCandidates asks chunksPerPath for every retrieved chunk of a path rather
// than a bounded best-n.
const allCandidates = math.MaxInt
