package index

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A symlinked file is not followed. os.ReadFile resolves the link, so without
// this a link named foo.go pointing outside the repository puts content the
// caller never offered into their index.
func TestWalkFilesDoesNotFollowSymlinks(t *testing.T) {
	outside := t.TempDir()
	secret := filepath.Join(outside, "secret.go")
	if err := os.WriteFile(secret, []byte("package p\n\nfunc Secret() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "real.go"), []byte("package p\n\nfunc Real() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(secret, filepath.Join(root, "linked.go")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	var seen []string
	err := WalkFiles(root, root, func(rel, content string) error {
		seen = append(seen, rel)
		if strings.Contains(content, "Secret") {
			t.Errorf("content from outside the root reached the index via %s", rel)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(seen) != 1 || seen[0] != "real.go" {
		t.Errorf("walked %v, want only [real.go]", seen)
	}
}

// Files past the cap are skipped before the read, so nothing downstream — the
// parse, the cover, or a single unbroken chunk holding the whole file — is
// handed an unbounded input.
func TestWalkFilesSkipsFilesOverTheSizeCap(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "small.go"), []byte("package p\n\nfunc Small() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	huge := append([]byte("package p\n\nvar Blob = `"), make([]byte, maxFileBytes)...)
	if err := os.WriteFile(filepath.Join(root, "huge.go"), append(huge, '`'), 0o644); err != nil {
		t.Fatal(err)
	}

	var seen []string
	if err := WalkFiles(root, root, func(rel, _ string) error {
		seen = append(seen, rel)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 1 || seen[0] != "small.go" {
		t.Errorf("walked %v, want only [small.go]", seen)
	}
}
