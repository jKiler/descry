package store

import (
	"database/sql"
	"errors"
	"fmt"

	"github.com/jKiler/descry/internal/core"

	_ "modernc.org/sqlite" // pure-Go driver, registers the "sqlite" driver name
)

// SchemaVersion is the on-disk table layout version. Bump it when the schema
// itself changes (columns, tables) — it's part of the reindex fingerprint.
const SchemaVersion = 1

// Fingerprint captures everything that, if changed, makes a stored index invalid
// and must trigger a full rebuild: the schema layout, a manual pipeline version
// bumped whenever a change improves stored-data quality (chunking, tokenization,
// graph, etc.), and the identities of the embedder and chunker in use.
type Fingerprint struct {
	Schema   int
	Pipeline int
	Embedder string
	Dim      int
	Chunker  string
}

func (f Fingerprint) String() string {
	return fmt.Sprintf("schema=%d pipeline=%d embedder=%s dim=%d chunker=%s",
		f.Schema, f.Pipeline, f.Embedder, f.Dim, f.Chunker)
}

// SQLiteStore persists chunks + vectors in a single SQLite file, and mirrors
// them in an in-memory MemStore so reads (Nearest/All/Len) stay fast and
// error-free. SQLite is the durable source of truth; memory is the query layer.
//
// Why mirror instead of querying SQLite on every Nearest? The Store interface's
// read methods don't return errors (they were shaped for the in-memory store),
// and at codebase scale a brute-force cosine scan in Go is already sub-10ms —
// so we load once on open and search in memory. SQLite just makes it durable.
// Implements store.Store — a drop-in for MemStore.
type SQLiteStore struct {
	db  *sql.DB
	mem *MemStore
}

const sqliteSchema = `
CREATE TABLE IF NOT EXISTS chunks (
	id         TEXT PRIMARY KEY,
	path       TEXT,
	start_line INTEGER,
	end_line   INTEGER,
	symbol     TEXT,
	content    TEXT,
	vector     BLOB
);
CREATE TABLE IF NOT EXISTS meta (
	key   TEXT PRIMARY KEY,
	value TEXT
);
CREATE TABLE IF NOT EXISTS lexicals (
	kind    INTEGER PRIMARY KEY,
	version INTEGER,
	chunks  INTEGER,
	data    BLOB
);
DROP TABLE IF EXISTS lexical;`

// lexicalVersion is the persisted lexical-index (BM25) blob version. It is
// independent of the on-disk schema: bump it when the BM25 encoding changes so a
// stale blob is ignored and rebuilt.
const lexicalVersion = 1

// Lexical index kinds: the retriever keeps two BM25 indexes, one over chunks
// and one over whole-file documents (see search.FileDocs). Each persists under
// its own kind.
const (
	LexicalChunks = 1
	LexicalFiles  = 2
)

// OpenSQLite opens (creating if needed) the database at path, ensures the
// schema, and loads any existing rows into the in-memory mirror.
func OpenSQLite(path string, fp Fingerprint) (*SQLiteStore, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec(sqliteSchema); err != nil {
		db.Close()
		return nil, err
	}
	if err := ensureFingerprint(db, fp); err != nil {
		db.Close()
		return nil, err
	}

	mem := NewMemStore()
	rows, err := db.Query(`SELECT id, path, start_line, end_line, symbol, content, vector FROM chunks`)
	if err != nil {
		db.Close()
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var (
			c   core.Chunk
			vec []byte
		)
		if err := rows.Scan(&c.ID, &c.Path, &c.StartLine, &c.EndLine, &c.Symbol, &c.Content, &vec); err != nil {
			db.Close()
			return nil, err
		}
		c.Vector = core.DecodeVec(vec)
		mem.Add(c)
	}
	if err := rows.Err(); err != nil {
		db.Close()
		return nil, err
	}
	return &SQLiteStore{db: db, mem: mem}, nil
}

// ensureFingerprint compares the stored index fingerprint with the current one.
// On a fresh db it records it; on a mismatch it clears the chunks so the caller
// re-indexes via the existing "if Len()==0" path. That's the seamless, automatic
// reindex on any version/config bump — no manual delete, no stale data.
func ensureFingerprint(db *sql.DB, fp Fingerprint) error {
	want := fp.String()

	var have string
	err := db.QueryRow(`SELECT value FROM meta WHERE key = 'fingerprint'`).Scan(&have)
	switch {
	case errors.Is(err, sql.ErrNoRows): // fresh db — stamp it
		_, err = db.Exec(`INSERT INTO meta(key, value) VALUES('fingerprint', ?)`, want)
		return err
	case err != nil:
		return err
	case have != want: // incompatible index — clear it and re-stamp
		if _, err := db.Exec(`DELETE FROM chunks`); err != nil {
			return err
		}
		// The lexical indexes are derived from the chunks, so they must not
		// outlive them.
		if _, err := db.Exec(`DELETE FROM lexicals`); err != nil {
			return err
		}
		_, err = db.Exec(`UPDATE meta SET value = ? WHERE key = 'fingerprint'`, want)
		return err
	}
	return nil // fingerprint matches — keep the existing index
}

const insertChunk = `INSERT OR REPLACE INTO chunks(id, path, start_line, end_line, symbol, content, vector)
VALUES(?, ?, ?, ?, ?, ?, ?)`

// Add persists one chunk — just a single-element batch.
func (s *SQLiteStore) Add(c core.Chunk) error {
	return s.AddBatch([]core.Chunk{c})
}

// AddBatch persists many chunks in ONE transaction, then mirrors them in memory
// once the commit succeeds. Batching turns N fsync-per-row commits into a single
// commit — the difference between seconds and milliseconds indexing a whole repo.
func (s *SQLiteStore) AddBatch(chunks []core.Chunk) error {
	if len(chunks) == 0 {
		return nil
	}
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	stmt, err := tx.Prepare(insertChunk)
	if err != nil {
		tx.Rollback()
		return err
	}
	defer stmt.Close()
	for _, c := range chunks {
		if _, err := stmt.Exec(c.ID, c.Path, c.StartLine, c.EndLine, c.Symbol, c.Content, core.EncodeVec(c.Vector)); err != nil {
			tx.Rollback()
			return err
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	// Mirror only after a successful commit, so memory matches disk.
	return s.mem.AddBatch(chunks)
}

// SetRerankMult tunes the quantized-search rerank pool (see MemStore).
func (s *SQLiteStore) SetRerankMult(m int) { s.mem.SetRerankMult(m) }

// Nearest, All, Len serve from the in-memory mirror (populated on open + add).
func (s *SQLiteStore) Nearest(vector []float32, topK int) []core.SearchResult {
	return s.mem.Nearest(vector, topK)
}
func (s *SQLiteStore) All() []core.Chunk { return s.mem.All() }
func (s *SQLiteStore) Len() int          { return s.mem.Len() }

// SaveLexical persists one serialized lexical (BM25) index under its kind
// (LexicalChunks or LexicalFiles), stamped with the current lexicalVersion and
// the chunk count it was built from. It replaces any previous blob of that kind.
func (s *SQLiteStore) SaveLexical(kind int, data []byte, chunks int) error {
	_, err := s.db.Exec(`INSERT OR REPLACE INTO lexicals(kind, version, chunks, data) VALUES(?, ?, ?, ?)`,
		kind, lexicalVersion, chunks, data)
	return err
}

// LoadLexical returns the persisted lexical index blob of the given kind and
// the chunk count it was built from. ok is false when there is no blob or it
// was written by a different lexicalVersion, so the caller rebuilds.
func (s *SQLiteStore) LoadLexical(kind int) (data []byte, chunks int, ok bool) {
	var ver int
	err := s.db.QueryRow(`SELECT version, chunks, data FROM lexicals WHERE kind = ?`, kind).Scan(&ver, &chunks, &data)
	if err != nil || ver != lexicalVersion {
		return nil, 0, false
	}
	return data, chunks, true
}

// Close closes the underlying database.
func (s *SQLiteStore) Close() error { return s.db.Close() }

var _ Store = (*SQLiteStore)(nil)
