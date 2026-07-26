// Package store holds indexed chunks and answers nearest-neighbor queries.
//
// MemStore (mem.go) is an in-memory slice with a brute-force cosine scan: O(n)
// per query, but simple and correct, and fast enough at codebase scale.
// SQLiteStore (sqlite.go) persists chunks + vectors to a single .db file and
// mirrors them in a MemStore for querying. Both satisfy the Store interface.
package store

import "github.com/jKiler/descry/internal/core"

// Store indexes chunks and finds the nearest ones to a query vector.
// (Both implementations also offer a single-chunk Add convenience, but the
// pipeline only ever inserts in batches, so the interface doesn't require it.)
type Store interface {
	AddBatch(chunks []core.Chunk) error // insert many at once (one transaction for SQLite)
	Nearest(vector []float32, topK int) []core.SearchResult
	All() []core.Chunk
	Len() int
}
