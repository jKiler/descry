package search

import (
	"testing"

	"github.com/jKiler/descry/internal/core"
)

// fixedStore returns a canned vector ranking, so a test can state "the vector
// retriever prefers this chunk" without a model.
type fixedStore struct{ ranked []core.SearchResult }

func (s *fixedStore) AddBatch([]core.Chunk) error { return nil }
func (s *fixedStore) All() []core.Chunk           { return nil }
func (s *fixedStore) Len() int                    { return len(s.ranked) }
func (s *fixedStore) Nearest(_ []float32, topK int) []core.SearchResult {
	return s.ranked[:min(topK, len(s.ranked))]
}

type nullEmbedder struct{}

func (nullEmbedder) Dim() int               { return 1 }
func (nullEmbedder) Embed(string) []float32 { return []float32{0} }
func (nullEmbedder) ID() string             { return "null" }

// TestSearchFilesShowsTheChunkTheRankersAgreeOn pins the choice of *which*
// chunk represents a file.
//
// One file, two chunks. The vector ranking puts the wrong one first; the
// lexical ranking puts the right one first and is the only ranker that has seen
// the query's terms at all. The file ranks either way — both chunks belong to
// it — so the only question is which chunk comes back, and the answer has to
// come from both rankers rather than from whichever list was collapsed first.
func TestSearchFilesShowsTheChunkTheRankersAgreeOn(t *testing.T) {
	wrong := core.Chunk{ID: "f#1", Path: "pkg/f.go", Symbol: "unrelated",
		Content: "func unrelated() { return }", StartLine: 1, EndLine: 3}
	right := core.Chunk{ID: "f#2", Path: "pkg/f.go", Symbol: "ParseQuantity",
		Content:   "func ParseQuantity(s string) (Quantity, error) { return parse(s) }",
		StartLine: 20, EndLine: 40}

	chunks := []core.Chunk{wrong, right}
	lex := NewBM25()
	lex.Index(chunks)
	h := NewHybrid(nullEmbedder{}, &fixedStore{ranked: []core.SearchResult{
		{Chunk: wrong, Score: 0.9}, // the vector ranking's pick for this file
		{Chunk: right, Score: 0.1},
	}}, lex)

	got := h.SearchFiles("ParseQuantity", 5)
	if len(got) == 0 {
		t.Fatal("no results")
	}
	if got[0].Chunk.Path != "pkg/f.go" {
		t.Fatalf("wrong file: %q", got[0].Chunk.Path)
	}
	if got[0].Chunk.ID != right.ID {
		t.Errorf("file represented by %q (%s); the lexical ranker put %q first and the vector ranking is indifferent",
			got[0].Chunk.ID, got[0].Chunk.Symbol, right.ID)
	}
}

// A file that only the whole-file lexical index finds has no chunk in either
// chunk ranking. It must still come back with something rather than a zero
// Chunk — losing the result entirely would be worse than showing a coarse one.
func TestSearchFilesKeepsFileOnlyMatches(t *testing.T) {
	indexed := core.Chunk{ID: "a#1", Path: "pkg/a.go", Content: "package a", StartLine: 1, EndLine: 1}
	other := core.Chunk{ID: "b#1", Path: "pkg/b.go", Symbol: "Rare",
		Content: "func Rare() {}", StartLine: 1, EndLine: 2}

	all := []core.Chunk{indexed, other}
	lex := NewBM25()
	lex.Index([]core.Chunk{indexed})
	h := NewHybrid(nullEmbedder{}, &fixedStore{ranked: []core.SearchResult{{Chunk: indexed}}}, lex)
	h.LexFile = NewBM25()
	h.LexFile.Index(FileDocs(all))

	for _, r := range h.SearchFiles("Rare", 5) {
		if r.Chunk.Path == "pkg/b.go" {
			if r.Chunk.Content == "" {
				t.Fatal("file-only match came back with an empty chunk")
			}
			return
		}
	}
	t.Fatal("the whole-file index found pkg/b.go but it is not in the results")
}

// SearchFilesN at n = 1 must be SearchFiles, exactly. The whole point of adding
// it is to measure wider results *against* the shipped one-chunk behaviour, and
// a comparison whose baseline arm is subtly a different retriever measures
// nothing. This has gone wrong before, because a
// metric moved underneath a comparison; this is the guard against a repeat.
func TestSearchFilesNAtOneMatchesSearchFiles(t *testing.T) {
	chunks := []core.Chunk{
		{ID: "a#1", Path: "a.go", Symbol: "Alpha", Content: "func Alpha() { quantity parse }"},
		{ID: "a#2", Path: "a.go", Symbol: "Alpha2", Content: "func Alpha2() { quantity }"},
		{ID: "a#3", Path: "a.go", Symbol: "Alpha3", Content: "func Alpha3() { other }"},
		{ID: "b#1", Path: "b.go", Symbol: "Beta", Content: "func Beta() { quantity quantity }"},
		{ID: "b#2", Path: "b.go", Symbol: "Beta2", Content: "func Beta2() { parse }"},
		{ID: "c#1", Path: "c.go", Symbol: "Gamma", Content: "func Gamma() { unrelated }"},
	}
	lex := NewBM25()
	lex.Index(chunks)
	ranked := make([]core.SearchResult, len(chunks))
	for i, c := range chunks {
		ranked[i] = core.SearchResult{Chunk: c, Score: float64(len(chunks) - i)}
	}
	build := func() *Hybrid { return NewHybrid(nullEmbedder{}, &fixedStore{ranked: ranked}, lex) }

	for _, q := range []string{"quantity", "parse", "Alpha", "unrelated quantity"} {
		one := build().SearchFiles(q, 10)
		many := build().SearchFilesN(q, 10, 1)
		if len(one) != len(many) {
			t.Fatalf("%q: %d results vs %d", q, len(one), len(many))
		}
		for i := range one {
			if one[i].Chunk.Path != many[i].Path {
				t.Errorf("%q rank %d: path %q vs %q", q, i, one[i].Chunk.Path, many[i].Path)
			}
			if one[i].Score != many[i].Score {
				t.Errorf("%q rank %d: score %v vs %v", q, i, one[i].Score, many[i].Score)
			}
			if len(many[i].Chunks) != 1 {
				t.Fatalf("%q rank %d: n=1 returned %d chunks", q, i, len(many[i].Chunks))
			}
			if one[i].Chunk.ID != many[i].Chunks[0].ID {
				t.Errorf("%q rank %d: chunk %q vs %q", q, i, one[i].Chunk.ID, many[i].Chunks[0].ID)
			}
		}
	}
}

// Raising n must widen files that have more retrieved chunks and change nothing
// else: same files, same order, same scores, and the first chunk of each file
// still the one n=1 would have shown.
func TestSearchFilesNWidensWithoutMovingTheFileRanking(t *testing.T) {
	chunks := []core.Chunk{
		{ID: "a#1", Path: "a.go", Symbol: "Alpha", Content: "func Alpha() { quantity parse }"},
		{ID: "a#2", Path: "a.go", Symbol: "Alpha2", Content: "func Alpha2() { quantity }"},
		{ID: "a#3", Path: "a.go", Symbol: "Alpha3", Content: "func Alpha3() { quantity other }"},
		{ID: "b#1", Path: "b.go", Symbol: "Beta", Content: "func Beta() { quantity quantity }"},
		{ID: "c#1", Path: "c.go", Symbol: "Gamma", Content: "func Gamma() { quantity unrelated }"},
	}
	lex := NewBM25()
	lex.Index(chunks)
	ranked := make([]core.SearchResult, len(chunks))
	for i, c := range chunks {
		ranked[i] = core.SearchResult{Chunk: c, Score: float64(len(chunks) - i)}
	}
	build := func() *Hybrid { return NewHybrid(nullEmbedder{}, &fixedStore{ranked: ranked}, lex) }

	one := build().SearchFilesN("quantity", 10, 1)
	three := build().SearchFilesN("quantity", 10, 3)
	if len(one) != len(three) {
		t.Fatalf("result count changed: %d -> %d", len(one), len(three))
	}
	widened := 0
	for i := range one {
		if one[i].Path != three[i].Path || one[i].Score != three[i].Score {
			t.Errorf("rank %d moved: %q/%v -> %q/%v", i, one[i].Path, one[i].Score, three[i].Path, three[i].Score)
		}
		if one[i].Chunks[0].ID != three[i].Chunks[0].ID {
			t.Errorf("rank %d: first chunk changed %q -> %q", i, one[i].Chunks[0].ID, three[i].Chunks[0].ID)
		}
		if len(three[i].Chunks) > 3 {
			t.Errorf("rank %d: n=3 returned %d chunks", i, len(three[i].Chunks))
		}
		if len(three[i].Chunks) > 1 {
			widened++
		}
	}
	if widened == 0 {
		t.Error("no file widened at n=3; the fixture cannot detect a regression")
	}
}
