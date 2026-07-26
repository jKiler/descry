package embed

import (
	"math"
	"testing"
)

// HashEmbedder behavior: bag-of-words hashing, L2-normalized, cosine == 1 with
// itself. Run: go test ./internal/embed
func TestCosine_IdenticalTextIsOne(t *testing.T) {
	h := NewHashEmbedder(256)
	v := h.Embed("user authentication and login handling")
	if got := Cosine(v, v); math.Abs(got-1.0) > 1e-6 {
		t.Errorf("cosine(v,v) = %v, want 1.0 (did you L2-normalize?)", got)
	}
}

func TestCosine_RelatedTextCloserThanUnrelated(t *testing.T) {
	h := NewHashEmbedder(512)
	a := h.Embed("user authentication login session")
	related := h.Embed("authentication login session token")
	unrelated := h.Embed("banana bread recipe with walnuts")

	simRelated := Cosine(a, related)
	simUnrelated := Cosine(a, unrelated)

	if !(simRelated > simUnrelated) {
		t.Errorf("related (%.3f) should score higher than unrelated (%.3f)", simRelated, simUnrelated)
	}
	if simUnrelated > 0.2 {
		t.Errorf("unrelated similarity unexpectedly high: %.3f", simUnrelated)
	}
}

func TestEmbed_Dimension(t *testing.T) {
	h := NewHashEmbedder(384)
	if v := h.Embed("hello"); len(v) != 384 {
		t.Errorf("embedding dim = %d, want 384", len(v))
	}
}

// Regression: Cosine on NON-normalized vectors. Embeddings from Embed are
// pre-normalized (|v|≈1), which hides an operator-precedence bug in the
// formula — dot/(|a|*|b|) must not be written dot/|a|*|b|.
func TestCosine_NonNormalizedVectors(t *testing.T) {
	if got := Cosine([]float32{3, 4}, []float32{3, 4}); math.Abs(got-1.0) > 1e-6 {
		t.Errorf("Cosine([3,4],[3,4]) = %v, want 1.0", got)
	}
	if got := Cosine([]float32{1, 0}, []float32{0, 5}); math.Abs(got) > 1e-6 {
		t.Errorf("orthogonal Cosine = %v, want 0", got)
	}
	if got := Cosine([]float32{1, 0}, []float32{1, 1}); math.Abs(got-0.70710678) > 1e-4 {
		t.Errorf("Cosine([1,0],[1,1]) = %v, want ~0.7071", got)
	}
	if got := Cosine([]float32{0, 0}, []float32{1, 1}); got != 0 {
		t.Errorf("zero-vector Cosine = %v, want 0 (no divide-by-zero)", got)
	}
}
