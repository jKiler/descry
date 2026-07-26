package store

import (
	"path/filepath"
	"testing"

	"github.com/jKiler/descry/internal/core"
)

var testFP = Fingerprint{Schema: 1, Pipeline: 1, Embedder: "test", Dim: 3, Chunker: "test"}

// The point of persistence: index once, close, reopen from disk, and the data
// (and search) is still there without re-indexing.
func TestSQLiteStore_RoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.db")

	s, err := OpenSQLite(path, testFP)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := s.Add(core.Chunk{ID: "a", Path: "a.go", Vector: []float32{1, 0, 0}}); err != nil {
		t.Fatal(err)
	}
	if err := s.Add(core.Chunk{ID: "b", Path: "b.go", Vector: []float32{0, 1, 0}}); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Reopen with the SAME fingerprint — nothing re-indexed.
	s2, err := OpenSQLite(path, testFP)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer s2.Close()

	if s2.Len() != 2 {
		t.Fatalf("after reopen Len = %d, want 2", s2.Len())
	}
	got := s2.Nearest([]float32{1, 0, 0}, 1)
	if len(got) != 1 || got[0].Chunk.ID != "a" {
		t.Errorf("Nearest top after reopen = %v, want chunk a", got)
	}
}

// A fingerprint change (new embedder, schema bump, or a pipeline version bump)
// must clear the index so the caller re-indexes — no stale data survives.
func TestSQLiteStore_FingerprintMismatchRebuilds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.db")

	s, err := OpenSQLite(path, testFP)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Add(core.Chunk{ID: "a", Path: "a.go", Vector: []float32{1, 0, 0}}); err != nil {
		t.Fatal(err)
	}
	s.Close()

	// Reopen with a bumped pipeline version — the old index must be gone.
	bumped := testFP
	bumped.Pipeline++
	s2, err := OpenSQLite(path, bumped)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()
	if s2.Len() != 0 {
		t.Errorf("after fingerprint change Len = %d, want 0 (index auto-cleared)", s2.Len())
	}
}

// The lexical index blob survives a round-trip, and a fingerprint change clears
// it along with the chunks (it's derived from them).
func TestSQLiteStore_LexicalPersistAndInvalidate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.db")

	s, err := OpenSQLite(path, testFP)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, ok := s.LoadLexical(LexicalChunks); ok {
		t.Error("fresh store should have no lexical blob")
	}
	blob := []byte("lexical-index-bytes")
	fileBlob := []byte("file-lexical-bytes")
	if err := s.SaveLexical(LexicalChunks, blob, 2); err != nil {
		t.Fatal(err)
	}
	if err := s.SaveLexical(LexicalFiles, fileBlob, 2); err != nil {
		t.Fatal(err)
	}
	s.Close()

	// Same fingerprint: both blobs survive, each under its own kind.
	s2, err := OpenSQLite(path, testFP)
	if err != nil {
		t.Fatal(err)
	}
	got, n, ok := s2.LoadLexical(LexicalChunks)
	if !ok || n != 2 || string(got) != string(blob) {
		t.Errorf("reload lexical = (%q, %d, %v), want (%q, 2, true)", got, n, ok, blob)
	}
	got, n, ok = s2.LoadLexical(LexicalFiles)
	if !ok || n != 2 || string(got) != string(fileBlob) {
		t.Errorf("reload file lexical = (%q, %d, %v), want (%q, 2, true)", got, n, ok, fileBlob)
	}
	s2.Close()

	// Fingerprint change: blobs are cleared with the chunks.
	bumped := testFP
	bumped.Pipeline++
	s3, err := OpenSQLite(path, bumped)
	if err != nil {
		t.Fatal(err)
	}
	defer s3.Close()
	if _, _, ok := s3.LoadLexical(LexicalChunks); ok {
		t.Error("fingerprint change must clear the lexical blob")
	}
	if _, _, ok := s3.LoadLexical(LexicalFiles); ok {
		t.Error("fingerprint change must clear the file lexical blob")
	}
}
