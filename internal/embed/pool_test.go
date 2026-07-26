package embed

import (
	"math"
	"testing"
)

// Mean-pooling behavior. Run: go test ./internal/embed -run MeanPool
//
// The model outputs one 384-dim vector PER TOKEN (last_hidden_state); a
// sentence embedding is the mask-weighted mean of those. Masked positions
// (padding) must not contribute — that's the whole point of weighting.

func TestMeanPool_MaskedMean(t *testing.T) {
	// 3 tokens, dim 2. Third token is masked out, so its values must be
	// ignored no matter how large: mean([1,2],[3,4]) = [2,3].
	hidden := []float32{
		1, 2,
		3, 4,
		100, 200,
	}
	got := meanPool(hidden, []int64{1, 1, 0}, 2)
	want := []float32{2, 3}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Errorf("pooled[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestMeanPool_AllMaskedNoNaN(t *testing.T) {
	// Degenerate all-zero mask: return a zero vector, not NaN (divide by zero).
	got := meanPool([]float32{1, 2, 3, 4}, []int64{0, 0}, 2)
	for i, x := range got {
		if x != 0 || math.IsNaN(float64(x)) {
			t.Errorf("pooled[%d] = %v, want 0", i, x)
		}
	}
}
