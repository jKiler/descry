// Package eval measures retrieval quality with the standard information-
// retrieval metrics: Recall@k, reciprocal rank, and MRR.
package eval

// toSet turns an id slice into a set for O(1) membership checks.
func toSet(ids []string) map[string]bool {
	s := make(map[string]bool, len(ids))
	for _, id := range ids {
		s[id] = true
	}
	return s
}

// RecallAtK: of the relevant ids, what fraction appear in the top-k retrieved.
func RecallAtK(retrieved, relevant []string, k int) float64 {
	rel := toSet(relevant)
	if len(rel) == 0 {
		return 0 // nothing to find; also avoids divide-by-zero
	}
	if k > len(retrieved) {
		k = len(retrieved)
	}
	if k < 0 {
		k = 0
	}
	found := make(map[string]bool)
	for _, id := range retrieved[:k] {
		if rel[id] {
			found[id] = true // distinct gold hits only, so duplicates don't over-count
		}
	}
	return float64(len(found)) / float64(len(rel))
}

// ReciprocalRank: 1/(rank of the first retrieved id that is relevant), or 0 if
// none are relevant. rank is 1-based. Averaged over a query set this is MRR
// (see EvaluateAtKs / EvaluateVerbose).
func ReciprocalRank(retrieved []string, relevant map[string]bool) float64 {
	for i, id := range retrieved {
		if relevant[id] {
			return 1.0 / float64(i+1)
		}
	}
	return 0
}
