package search

import (
	"testing"

	"github.com/jKiler/descry/internal/core"
)

func persistDocs() []core.Chunk {
	return []core.Chunk{
		{ID: "a", Path: "internal/search/bm25.go", Symbol: "Search", Content: "authentication middleware validates the jwt token cache"},
		{ID: "b", Path: "internal/store/sqlite.go", Symbol: "Nearest", Content: "vector store nearest neighbor cosine cache"},
		{ID: "c", Path: "readme.md", Content: "a recipe for banana bread and walnuts"},
	}
}

// A decoded index must answer identically to the one it was encoded from.
func TestBM25EncodeDecodeRoundTrip(t *testing.T) {
	docs := persistDocs()
	orig := NewBM25()
	orig.Index(docs)

	blob := orig.Encode()
	if len(blob) == 0 {
		t.Fatal("Encode returned no data")
	}
	got, ok := DecodeBM25(blob, docs)
	if !ok {
		t.Fatal("DecodeBM25 rejected a freshly-encoded blob")
	}

	for _, q := range []string{"jwt token", "vector cache", "nearest", "search", "banana"} {
		a := orig.Search(q, 10)
		b := got.Search(q, 10)
		if len(a) != len(b) {
			t.Fatalf("q=%q: decoded returned %d results, want %d", q, len(b), len(a))
		}
		for i := range a {
			if a[i].Chunk.ID != b[i].Chunk.ID || a[i].Score != b[i].Score {
				t.Errorf("q=%q rank %d: decoded (%s,%v) != original (%s,%v)",
					q, i, b[i].Chunk.ID, b[i].Score, a[i].Chunk.ID, a[i].Score)
			}
		}
	}
}

// A blob must be rejected when the chunks differ from what it was built for.
func TestDecodeBM25RejectsMismatch(t *testing.T) {
	docs := persistDocs()
	orig := NewBM25()
	orig.Index(docs)
	blob := orig.Encode()

	// Fewer chunks than encoded.
	if _, ok := DecodeBM25(blob, docs[:2]); ok {
		t.Error("decode should reject a different chunk count")
	}
	// Same count, different ids (id hash mismatch).
	altered := persistDocs()
	altered[0].ID = "z"
	if _, ok := DecodeBM25(blob, altered); ok {
		t.Error("decode should reject a different chunk-id set")
	}
	// Truncated blob.
	if _, ok := DecodeBM25(blob[:len(blob)/2], docs); ok {
		t.Error("decode should reject a truncated blob")
	}
	// Garbage.
	if _, ok := DecodeBM25([]byte{0xff, 0x00, 0x01}, docs); ok {
		t.Error("decode should reject garbage")
	}
}
