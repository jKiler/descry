// CachedEmbedder: a persistent read-through vector cache around any Embedder.
//
// The ONNX embedder costs ~0.5s per real code chunk, and the index fingerprint
// deliberately throws the whole index away on any pipeline/schema change — so
// every quality bump used to re-pay the full embedding cost even though almost
// no chunk text changed. Caching vectors by (embedder id, sha256 of the exact
// embed text) makes those rebuilds nearly free: only genuinely new or edited
// chunks hit the model. The cache lives in its own SQLite file, outside the
// index, precisely so fingerprint-driven index clears don't clear it.
package embed

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"

	"github.com/jKiler/descry/internal/core"

	_ "modernc.org/sqlite" // pure-Go driver, registers the "sqlite" driver name
)

// CachedEmbedder wraps an Embedder, serving repeated texts from SQLite.
// It reports the inner embedder's ID/Dim, so it's invisible to the index
// fingerprint. Safe for concurrent use (the pool in index.embedChunks):
// database/sql serializes access through one connection.
type CachedEmbedder struct {
	inner Embedder
	db    *sql.DB
}

const cacheSchema = `
CREATE TABLE IF NOT EXISTS vectors (
	embedder TEXT NOT NULL,
	hash     TEXT NOT NULL,
	vector   BLOB NOT NULL,
	PRIMARY KEY (embedder, hash)
);`

// NewCachedEmbedder opens (creating if needed) the cache database at path and
// wraps inner with it.
func NewCachedEmbedder(inner Embedder, path string) (*CachedEmbedder, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open embed cache: %w", err)
	}
	// One connection: writes from concurrent embed workers queue up instead of
	// fighting for the write lock (SQLITE_BUSY). Lookups are sub-ms; the model
	// call they replace is ~500ms, so serialization here costs nothing.
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(cacheSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("embed cache schema: %w", err)
	}
	return &CachedEmbedder{inner: inner, db: db}, nil
}

// ID reports the inner embedder's identity — the cache never changes vectors,
// so it must not change the fingerprint either.
func (c *CachedEmbedder) ID() string { return c.inner.ID() }

// Dim reports the inner embedder's dimensionality.
func (c *CachedEmbedder) Dim() int { return c.inner.Dim() }

// Embed returns the cached vector for text, or computes and stores it.
func (c *CachedEmbedder) Embed(text string) []float32 {
	sum := sha256.Sum256([]byte(text))
	key := hex.EncodeToString(sum[:])

	var blob []byte
	err := c.db.QueryRow(`SELECT vector FROM vectors WHERE embedder = ? AND hash = ?`,
		c.inner.ID(), key).Scan(&blob)
	if err == nil && len(blob) == c.inner.Dim()*4 {
		return core.DecodeVec(blob)
	}

	v := c.inner.Embed(text)
	if v != nil {
		// Best-effort: a failed write just means a cache miss next time.
		c.db.Exec(`INSERT OR REPLACE INTO vectors(embedder, hash, vector) VALUES(?, ?, ?)`,
			c.inner.ID(), key, core.EncodeVec(v))
	}
	return v
}

// Close closes the cache database.
func (c *CachedEmbedder) Close() error { return c.db.Close() }

var _ Embedder = (*CachedEmbedder)(nil)
