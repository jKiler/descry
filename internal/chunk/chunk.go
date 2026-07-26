// Package chunk splits file content into retrievable pieces.
//
// LineChunker (line.go) splits on blank lines into "paragraph" chunks — the
// fallback for files without a known grammar. ASTChunker (ast.go) splits Go
// source on real func/type/method boundaries and falls back to LineChunker for
// non-Go or unparseable files. Both satisfy the Chunker interface, so either
// can drop into the pipeline without touching callers.
package chunk

import "github.com/jKiler/descry/internal/core"

// Chunker turns one file's content into zero or more chunks.
type Chunker interface {
	Chunk(path, content string) []core.Chunk
	// ID identifies the chunking strategy. Changing how chunks are cut changes
	// the stored data, so it feeds the reindex fingerprint.
	ID() string
}
