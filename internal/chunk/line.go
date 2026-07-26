package chunk

import (
	"fmt"
	"strings"

	"github.com/jKiler/descry/internal/core"
)

// LineChunker splits content into chunks separated by blank lines.
type LineChunker struct {
	// MaxLines, if > 0, caps chunk size; oversized paragraphs are windowed.
	MaxLines int
}

// NewLineChunker returns a paragraph chunker with sensible defaults.
func NewLineChunker() *LineChunker { return &LineChunker{MaxLines: 0} }

// ID identifies the chunking strategy for the reindex fingerprint.
func (c *LineChunker) ID() string { return "line" }

// Chunk splits content on blank-line boundaries.
//
// Behavior (see line_test.go):
//   - Split content into groups of consecutive non-blank lines.
//   - Each group becomes one core.Chunk.
//   - StartLine/EndLine are 1-based and inclusive.
//   - ID is fmt.Sprintf("%s#%d", path, StartLine).
//   - Content is the group's lines re-joined with "\n" (no trailing newline).
//   - Leading/trailing blank lines and blank-only gaps produce no chunks.
func (c *LineChunker) Chunk(path, content string) []core.Chunk {
	var chunks []core.Chunk
	currLine := 1
	startLine := 1
	chunkContent := ""

	// flush emits the currently-accumulated chunk (if any) ending at the
	// previous line, then resets the accumulator.
	flush := func() {
		if chunkContent == "" {
			return
		}
		chunks = append(chunks, core.Chunk{
			ID:        fmt.Sprintf("%s#%d", path, startLine),
			Path:      path,
			StartLine: startLine,
			EndLine:   currLine - 1,
			Content:   strings.Trim(chunkContent, "\n"),
		})
		chunkContent = ""
	}

	for line := range strings.Lines(content) {
		// TrimSpace catches tab-only and CRLF blank lines, not just " \n".
		if strings.TrimSpace(line) == "" {
			flush()
			startLine = currLine + 1
		} else {
			chunkContent += line // keep the original line — preserves indentation
		}
		currLine++
	}
	flush() // final chunk (no trailing blank line to trigger it)
	return chunks
}

var _ Chunker = (*LineChunker)(nil)
