package search

import (
	"cmp"
	"maps"
	"slices"
)

// Reciprocal Rank Fusion.
//
// Vector search and BM25 produce scores on completely different scales, so they
// can't simply be added. RRF sidesteps this by combining *rank positions*: a
// document's fused score is the sum over lists of 1/(k + rank), where rank is
// 1-based. Robust and parameter-light.

// FusedResult is an id paired with its fused RRF score. The score is the real
// Σ 1/(k+rank) confidence, so callers (Hybrid.Search) can surface it instead of
// inventing a rank-derived placeholder. Scores are only comparable within one
// fusion call — RRF has no absolute scale.
type FusedResult struct {
	ID    string
	Score float64
}

// ReciprocalRankFusion merges several ranked ID lists into one.
//
//	lists:   each inner slice is IDs in rank order (best first).
//	k:       the RRF constant (common default 60). Larger k flattens the contribution of top ranks.
//	weights: optional per-list weight; if nil or wrong length, treat all as 1.0.
//
// Behavior: for every id, score += weight[i] * 1/(k + rank) across the lists it
// appears in (rank is 1-based). Return results sorted by descending fused score,
// ties broken by first appearance for determinism.
func ReciprocalRankFusion(lists [][]string, k float64, weights []float64) []FusedResult {
	return ReciprocalRankFusionAlpha(lists, k, weights, 1.0)
}

// ReciprocalRankFusionAlpha is ReciprocalRankFusion with a consensus dial.
//
// Plain RRF sums each list's 1/(k+rank) credit, which carries a consensus bias:
// a document ranked ~10th by every list outscores one ranked 3rd by a single
// list — even though a strong single-signal match is usually the better answer.
// Alpha blends the two views: a document's score is its best single-list
// contribution plus alpha times the rest.
//
//	alpha = 1: plain RRF (pure sum).
//	alpha = 0: pure max — each document scores by its best list only.
//
// Between the extremes, single-list excellence leads and agreement still adds.
func ReciprocalRankFusionAlpha(lists [][]string, k float64, weights []float64, alpha float64) []FusedResult {
	// nil OR wrong-length weights -> treat every list as weight 1.0.
	if len(weights) != len(lists) {
		weights = make([]float64, len(lists))
		for i := range weights {
			weights[i] = 1.0
		}
	}

	sums := make(map[string]float64)  // id -> total contribution across lists
	bests := make(map[string]float64) // id -> best single-list contribution
	firstSeen := make(map[string]int) // id -> first position seen (tie-break)
	for i, list := range lists {
		for rank, id := range list {
			c := weights[i] * (1.0 / (k + float64(rank+1)))
			sums[id] += c
			bests[id] = max(bests[id], c)
			if _, ok := firstSeen[id]; !ok {
				firstSeen[id] = len(firstSeen)
			}
		}
	}
	scores := make(map[string]float64, len(sums))
	for id, sum := range sums {
		scores[id] = bests[id] + alpha*(sum-bests[id])
	}

	ids := slices.Collect(maps.Keys(scores))
	slices.SortFunc(ids, func(a, b string) int {
		if scores[a] != scores[b] {
			return cmp.Compare(scores[b], scores[a]) // higher score first
		}
		return cmp.Compare(firstSeen[a], firstSeen[b]) // stable tie-break
	})

	fused := make([]FusedResult, len(ids))
	for i, id := range ids {
		fused[i] = FusedResult{ID: id, Score: scores[id]}
	}
	return fused
}
