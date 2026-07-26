package embed

import (
	"fmt"
	"hash/fnv"
	"strings"
	"unicode"
)

// HashEmbedder is a deterministic, dependency-free embedder: it hashes each word
// into one of D buckets, counts occurrences, and L2-normalizes the result, so
// similar wording produces a similar vector. It captures lexical overlap only —
// no semantics — but requires no model, which makes it a handy lightweight
// embedder for tests and offline use. For semantic search, use
// NewSemanticEmbedder.
type HashEmbedder struct {
	D int
}

// NewHashEmbedder returns a hashing embedder with the given dimension.
func NewHashEmbedder(dim int) *HashEmbedder { return &HashEmbedder{D: dim} }

// Dim reports the vector dimension.
func (h *HashEmbedder) Dim() int { return h.D }

// ID identifies this embedder (kind + dimension) for the reindex fingerprint.
func (h *HashEmbedder) ID() string { return fmt.Sprintf("hash-%d", h.D) }

// Embed produces the L2-normalized bag-of-words hashing vector for text: it
// lowercases the text, splits on non-letter/digit runs, and for each word adds 1
// to bucket fnv32(word) % D.
func (h *HashEmbedder) Embed(text string) []float32 {
	vector := make([]float32, h.D)
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	for _, word := range words {
		hasher := fnv.New32a()
		_, _ = hasher.Write([]byte(word)) // hash.Hash.Write never returns an error
		bucket := hasher.Sum32() % uint32(h.D)
		vector[bucket] += 1
	}
	return l2norm(vector)
}

var _ Embedder = (*HashEmbedder)(nil)
