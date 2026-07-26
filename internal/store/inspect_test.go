package store

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/jKiler/descry/internal/core"
)

func testFingerprint() Fingerprint {
	return Fingerprint{Schema: SchemaVersion, Pipeline: 1, Embedder: "test", Dim: 3, Chunker: "line"}
}

func TestInspectReportsFingerprintAndChunkCount(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.db")
	fp := testFingerprint()
	s, err := OpenSQLite(path, fp)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := s.Add(core.Chunk{ID: "a", Path: "a.go", Content: "x", Vector: []float32{1, 0, 0}}); err != nil {
		t.Fatalf("add: %v", err)
	}
	s.Close()

	info, err := Inspect(path)
	if err != nil {
		t.Fatalf("inspect: %v", err)
	}
	if info.Fingerprint != fp.String() {
		t.Errorf("fingerprint = %q, want %q", info.Fingerprint, fp.String())
	}
	if info.Chunks != 1 {
		t.Errorf("chunks = %d, want 1", info.Chunks)
	}
}

// TestInspectNeverMutatesTheIndex is the safety property that justifies Inspect
// existing at all: OpenSQLite *deletes* the chunks when the fingerprint has
// moved on, so a diagnostic built on it would destroy the very index the user
// is asking about. Inspect must only read, however often it is called.
func TestInspectNeverMutatesTheIndex(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.db")
	s, err := OpenSQLite(path, testFingerprint())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := s.Add(core.Chunk{ID: "a", Path: "a.go", Content: "x", Vector: []float32{1, 0, 0}}); err != nil {
		t.Fatalf("add: %v", err)
	}
	s.Close()

	// Inspect the index as a *newer* descry would: nothing about the call
	// carries the current fingerprint, so nothing can invalidate anything.
	for i := 0; i < 2; i++ {
		info, err := Inspect(path)
		if err != nil {
			t.Fatalf("inspect: %v", err)
		}
		if info.Chunks != 1 {
			t.Fatalf("inspect pass %d: chunks = %d, want 1 — inspection destroyed the index", i, info.Chunks)
		}
	}
}

// TestInspectHandlesURISyntaxInPath pins the DSN escaping. The driver opens
// READWRITE|CREATE and is restrained only by "mode=ro" in the URI, so an
// unescaped '#' or '?' in a repository path would truncate the filename SQLite
// sees, drop the mode parameter, and let an inspection *create* a database at
// the truncated path — the exact opposite of read-only.
func TestInspectHandlesURISyntaxInPath(t *testing.T) {
	for _, name := range []string{"a#b", "a?b", "a%b", "a b"} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			dir := filepath.Join(root, name)
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(dir, "index.db")
			s, err := OpenSQLite(path, testFingerprint())
			if err != nil {
				t.Fatalf("open: %v", err)
			}
			if err := s.Add(core.Chunk{ID: "a", Path: "a.go", Content: "x", Vector: []float32{1, 0, 0}}); err != nil {
				t.Fatalf("add: %v", err)
			}
			s.Close()

			info, err := Inspect(path)
			if err != nil {
				t.Fatalf("inspect: %v", err)
			}
			if info.Chunks != 1 {
				t.Errorf("chunks = %d, want 1 — the DSN did not reach the real file", info.Chunks)
			}
			// A truncated filename would have created a database beside the
			// directory instead of reading the one inside it.
			entries, err := os.ReadDir(root)
			if err != nil {
				t.Fatal(err)
			}
			if len(entries) != 1 {
				var got []string
				for _, e := range entries {
					got = append(got, e.Name())
				}
				t.Errorf("inspection created files: %v", got)
			}
		})
	}
}

func TestInspectMissingFileIsNotExist(t *testing.T) {
	_, err := Inspect(filepath.Join(t.TempDir(), "absent.db"))
	if !os.IsNotExist(err) {
		t.Errorf("err = %v, want an IsNotExist error so callers can offer `descry index`", err)
	}
}
