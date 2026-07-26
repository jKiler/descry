// Package embed turns text into a fixed-length vector (an "embedding") so that
// two pieces of text with similar meaning produce vectors pointing in a similar
// direction, measured by cosine similarity.
//
// The package ships two embedders behind a common interface:
//
//   - HashEmbedder — a deterministic, dependency-free bag-of-words hashing
//     embedder. It captures lexical overlap only, but needs no model download,
//     which makes it useful for tests and offline use.
//   - the semantic embedder (NewSemanticEmbedder) — all-MiniLM-L6-v2 run on
//     ONNX Runtime, the production default. See ort.go for the embedder and
//     ortlib.go for how the runtime library is provisioned.
//
// Because every consumer depends only on the Embedder interface, swapping
// implementations is a one-line change.
package embed

import "math"

// Embedder maps text to a vector of fixed dimension.
//
// Embed must be safe for concurrent use: the indexer fans it out across
// goroutines (index.embedChunks). HashEmbedder allocates fresh state per call;
// the ONNX embedders' shared model is read-only during inference (verified -race).
type Embedder interface {
	Dim() int
	Embed(text string) []float32
	// ID uniquely identifies this embedder and its config. Changing it
	// invalidates a stored index (different vector space), so it feeds the
	// reindex fingerprint.
	ID() string
}

// Cosine returns the cosine similarity of a and b in [-1, 1].
// Returns 0 if either vector has zero magnitude (or the lengths differ).
func Cosine(a, b []float32) float64 {
	magA, magB := mag(a), mag(b)
	if magA == 0 || magB == 0 {
		return 0
	}
	return float64(dot(a, b)) / (float64(magA) * float64(magB))
}

// l2norm scales v to unit length in place and returns it. An all-zero vector is
// returned unchanged (no divide-by-zero).
func l2norm(v []float32) []float32 {
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	norm := math.Sqrt(sumSq)
	if norm == 0 {
		return v
	}
	for i, x := range v {
		v[i] = float32(float64(x) / norm)
	}
	return v
}

func dot(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func mag(v []float32) float32 {
	var sum float32
	for i := range v {
		sum += v[i] * v[i]
	}
	return float32(math.Sqrt(float64(sum)))
}
