package eval

// Labeled is one evaluation query paired with the ids that *should* be retrieved
// (the "gold" set). In descry' CLI the ids are file paths, loaded from JSON.
type Labeled struct {
	Query    string   `json:"query"`
	Relevant []string `json:"relevant"`
}

// Retriever returns ranked ids for a query, best first. Passing this as a
// function (rather than importing the search package) keeps eval dependency-free
// and reusable — you can score any retriever, current or future.
type Retriever func(query string, k int) []string

// Report is the aggregate result of an evaluation run.
type Report struct {
	Queries   int
	K         int
	RecallAtK float64
	MRR       float64
}

// MultiReport is Report generalized to several recall cutoffs (R@5, R@7, R@10,
// R@15, R@20, MRR in one pass).
type MultiReport struct {
	Queries int
	Ks      []int
	Recall  []float64 // Recall[i] is Recall@Ks[i]
	MRR     float64
}

// EvaluateAtKs retrieves once per query (at the deepest cutoff) and scores
// recall at every k in ks, plus MRR over the full ranking. ks must be sorted
// ascending; the retriever is asked for max(ks) results.
func EvaluateAtKs(r Retriever, set []Labeled, ks []int) MultiReport {
	rep := MultiReport{Queries: len(set), Ks: ks, Recall: make([]float64, len(ks))}
	if len(set) == 0 || len(ks) == 0 {
		return rep
	}
	maxK := ks[len(ks)-1]
	for _, q := range set {
		ranked := r(q.Query, maxK)
		for i, k := range ks {
			rep.Recall[i] += RecallAtK(ranked, q.Relevant, k)
		}
		rel := make(map[string]bool, len(q.Relevant))
		for _, id := range q.Relevant {
			rel[id] = true
		}
		rep.MRR += ReciprocalRank(ranked, rel)
	}
	n := float64(len(set))
	for i := range rep.Recall {
		rep.Recall[i] /= n
	}
	rep.MRR /= n
	return rep
}

// QueryResult is one query's outcome: the 1-based rank of the first relevant
// id in the ranking, or 0 if none made the cut.
type QueryResult struct {
	Query string
	Rank  int
}

// EvaluateVerbose is Evaluate plus the per-query ranks, so a regression can be
// traced to the specific queries that moved instead of guessed from averages.
func EvaluateVerbose(r Retriever, set []Labeled, k int) (Report, []QueryResult) {
	if len(set) == 0 {
		return Report{K: k}, nil
	}
	var recallSum, rrSum float64
	perQuery := make([]QueryResult, 0, len(set))
	for _, q := range set {
		ranked := r(q.Query, k)
		recallSum += RecallAtK(ranked, q.Relevant, k)

		rel := make(map[string]bool, len(q.Relevant))
		for _, id := range q.Relevant {
			rel[id] = true
		}
		rr := ReciprocalRank(ranked, rel)
		rrSum += rr
		rank := 0
		if rr > 0 {
			rank = int(1/rr + 0.5)
		}
		perQuery = append(perQuery, QueryResult{Query: q.Query, Rank: rank})
	}
	n := float64(len(set))
	return Report{
		Queries:   len(set),
		K:         k,
		RecallAtK: recallSum / n,
		MRR:       rrSum / n,
	}, perQuery
}
