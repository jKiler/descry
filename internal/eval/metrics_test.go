package eval

import (
	"math"
	"testing"
)

// Recall@k metric.
func TestRecallAtK(t *testing.T) {
	retrieved := []string{"a", "b", "c", "d"}
	relevant := []string{"b", "x"} // only b is retrievable, and it's in top-3
	if got := RecallAtK(retrieved, relevant, 3); math.Abs(got-0.5) > 1e-9 {
		t.Errorf("RecallAtK = %v, want 0.5", got)
	}
	if got := RecallAtK(retrieved, relevant, 1); got != 0 {
		t.Errorf("RecallAtK@1 = %v, want 0 (b not in top-1)", got)
	}
}

func TestReciprocalRank(t *testing.T) {
	rr := ReciprocalRank([]string{"a", "b", "c"}, map[string]bool{"b": true})
	if math.Abs(rr-0.5) > 1e-9 {
		t.Errorf("ReciprocalRank = %v, want 0.5", rr)
	}
	if rr := ReciprocalRank([]string{"a", "b"}, map[string]bool{"z": true}); rr != 0 {
		t.Errorf("ReciprocalRank with no relevant hit = %v, want 0", rr)
	}
}
