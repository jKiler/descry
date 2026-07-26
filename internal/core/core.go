// Package core holds the shared domain types used across descry.
//
// Keeping these in one dependency-free package avoids import cycles:
// every other package imports core, core imports nothing.
package core

// Chunk is the unit of retrieval: a slice of a file (a function, a type, a
// markdown section, or a blank-line-delimited paragraph). Each chunk gets an
// embedding vector and is stored for search.
type Chunk struct {
	ID        string    // stable id, e.g. "path#startLine"
	Path      string    // source file, relative to index root
	StartLine int       // 1-based, inclusive
	EndLine   int       // 1-based, inclusive
	Symbol    string    // function/class/section name, "" if none
	Content   string    // the raw text of the chunk
	Vector    []float32 // filled in after embedding; nil until then
}

// SearchResult pairs a chunk with the score a retriever gave it.
// Higher Score = more relevant. Different retrievers use different scales,
// which is why rank fusion (see internal/search) merges by rank, not raw score.
type SearchResult struct {
	Chunk Chunk
	Score float64
}
