package chunk

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// IDFor builds a chunk's stable identity: its file, plus a hash of its content.
//
// Keying on content rather than on the start line is what makes an id survive
// edits elsewhere in the file. With `path#startLine`, adding one import at the
// top renumbers every chunk below it, so an incremental index re-inserts a
// whole file's chunks under new ids and an evaluation diff between two runs
// cannot tell a chunk that moved from a chunk that changed. With a content
// hash, only the chunks whose text actually changed get new ids.
//
// The store keys chunks by id (`INSERT OR REPLACE`), so identical content in the
// same file — two `pass` bodies, two identical getters — must not collide, or
// one would silently overwrite the other. seen tracks the ids already issued for
// this file and disambiguates a repeat with an occurrence suffix.
func IDFor(path, content string, seen map[string]int) string {
	sum := sha256.Sum256([]byte(content))
	id := path + "#" + hex.EncodeToString(sum[:6])
	if seen == nil {
		return id
	}
	n := seen[id]
	seen[id] = n + 1
	if n == 0 {
		return id
	}
	return fmt.Sprintf("%s~%d", id, n)
}
