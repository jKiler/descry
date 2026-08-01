package chunk

import (
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
//   - ID is content-keyed (see IDFor), not line-keyed.
//   - Content is the group's lines re-joined with "\n" (no trailing newline).
//   - Leading/trailing blank lines and blank-only gaps produce no chunks.
func (c *LineChunker) Chunk(path, content string) []core.Chunk {
	var chunks []core.Chunk
	seen := map[string]int{}
	currLine := 1
	startLine := 1
	// The chunk under construction is a half-open byte range into content, not an
	// accumulated string. Appending line by line is quadratic — each += copies
	// everything gathered so far — and this chunker is what handles JSON, YAML and
	// every file whose grammar failed, exactly the inputs that arrive as one
	// unbroken run of thousands of lines.
	chunkLen := 0
	// Byte offsets are tracked alongside lines so the structural-integrity check
	// can see exactly where this chunker cuts. Without them it would compare
	// against zero and pass vacuously — which is the opposite of the truth, since
	// blank-line splitting cuts through function bodies constantly.
	pos, startByte := 0, 0

	// flush emits the currently-accumulated chunk (if any) ending at the
	// previous line, then resets the accumulator.
	flush := func() {
		if chunkLen == 0 {
			return
		}
		raw := content[startByte : startByte+chunkLen]
		text := strings.Trim(raw, "\n")
		chunks = append(chunks, core.Chunk{
			ID:        IDFor(path, text, seen),
			Path:      path,
			StartLine: startLine,
			EndLine:   currLine - 1,
			Content:   text,
			StartByte: startByte,
			EndByte:   startByte + chunkLen,
		})
		chunkLen = 0
	}

	for line := range strings.Lines(content) {
		// TrimSpace catches tab-only and CRLF blank lines, not just " \n".
		if strings.TrimSpace(line) == "" {
			flush()
			startLine = currLine + 1
			startByte = pos + len(line)
		} else {
			if chunkLen == 0 {
				startByte = pos
			}
			chunkLen += len(line) // the slice keeps the original line verbatim
		}
		pos += len(line)
		currLine++
	}
	flush() // final chunk (no trailing blank line to trigger it)
	return chunks
}

var _ Chunker = (*LineChunker)(nil)
