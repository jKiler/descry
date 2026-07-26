package search

import "github.com/jKiler/descry/internal/tokenize"

// tokenizeShim lets the parity test tokenize exactly as BM25.Index does.
func tokenizeShim(s string) []string { return tokenize.Tokenize(s) }
