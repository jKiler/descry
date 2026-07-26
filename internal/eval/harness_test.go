package eval

import (
	"math"
	"reflect"
	"testing"
)

// EvaluateVerbose aggregates Recall@k and MRR over a query set.
func TestEvaluateVerbose(t *testing.T) {
	set := []Labeled{
		{Query: "q1", Relevant: []string{"a"}},
		{Query: "q2", Relevant: []string{"b"}},
	}
	retrieve := func(q string, k int) []string {
		switch q {
		case "q1":
			return []string{"a", "x"} // gold at rank 1
		case "q2":
			return []string{"y", "b"} // gold at rank 2
		}
		return nil
	}

	got, perQuery := EvaluateVerbose(retrieve, set, 10)
	if want := []QueryResult{{Query: "q1", Rank: 1}, {Query: "q2", Rank: 2}}; !reflect.DeepEqual(perQuery, want) {
		t.Errorf("perQuery = %v, want %v", perQuery, want)
	}
	if got.Queries != 2 {
		t.Errorf("Queries = %d, want 2", got.Queries)
	}
	if math.Abs(got.RecallAtK-1.0) > 1e-9 {
		t.Errorf("Recall@10 = %v, want 1.0 (both gold found in top 10)", got.RecallAtK)
	}
	if math.Abs(got.MRR-0.75) > 1e-9 { // (1/1 + 1/2) / 2
		t.Errorf("MRR = %v, want 0.75", got.MRR)
	}
}

// EvaluateAtKs scores one retrieval pass at several cutoffs: a gold at rank 3
// counts for R@5 but not R@1, and MRR still reflects the exact rank.
func TestEvaluateAtKs(t *testing.T) {
	set := []Labeled{
		{Query: "q1", Relevant: []string{"a"}}, // gold at rank 1
		{Query: "q2", Relevant: []string{"b"}}, // gold at rank 3
	}
	retrieve := func(q string, k int) []string {
		if k != 5 {
			t.Errorf("retriever asked for k=%d, want 5 (deepest cutoff)", k)
		}
		switch q {
		case "q1":
			return []string{"a", "x", "y"}
		case "q2":
			return []string{"x", "y", "b"}
		}
		return nil
	}

	got := EvaluateAtKs(retrieve, set, []int{1, 5})
	if got.Queries != 2 {
		t.Errorf("Queries = %d, want 2", got.Queries)
	}
	if math.Abs(got.Recall[0]-0.5) > 1e-9 { // only q1's gold is in the top 1
		t.Errorf("Recall@1 = %v, want 0.5", got.Recall[0])
	}
	if math.Abs(got.Recall[1]-1.0) > 1e-9 {
		t.Errorf("Recall@5 = %v, want 1.0", got.Recall[1])
	}
	if math.Abs(got.MRR-(1.0+1.0/3)/2) > 1e-9 {
		t.Errorf("MRR = %v, want %v", got.MRR, (1.0+1.0/3)/2)
	}
}
