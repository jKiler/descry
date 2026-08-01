package chunk

import "testing"

// LineChunker behavior. Run: go test ./internal/chunk
func TestLineChunker_SplitsOnBlankLines(t *testing.T) {
	content := "line one\nline two\n\nsecond para\n\n\nthird para\n"
	//          1        2          4                        7
	got := NewLineChunker().Chunk("a.txt", content)

	if len(got) != 3 {
		t.Fatalf("want 3 chunks, got %d: %+v", len(got), got)
	}

	if got[0].Content != "line one\nline two" {
		t.Errorf("chunk0 content = %q", got[0].Content)
	}
	if got[0].StartLine != 1 || got[0].EndLine != 2 {
		t.Errorf("chunk0 lines = %d-%d, want 1-2", got[0].StartLine, got[0].EndLine)
	}
	if got[1].StartLine != 4 || got[1].Content != "second para" {
		t.Errorf("chunk1 = %d %q, want start 4 'second para'", got[1].StartLine, got[1].Content)
	}
	if got[2].StartLine != 7 || got[2].Content != "third para" {
		t.Errorf("chunk2 = %d %q, want start 7 'third para'", got[2].StartLine, got[2].Content)
	}
	if got[0].Path != "a.txt" {
		t.Errorf("path = %q, want a.txt", got[0].Path)
	}
	// Ids are content-keyed, so they survive edits elsewhere in the file.
	if want := IDFor("a.txt", got[0].Content, nil); got[0].ID != want {
		t.Errorf("id = %q, want %q", got[0].ID, want)
	}
}

func TestLineChunker_EmptyInput(t *testing.T) {
	if got := NewLineChunker().Chunk("x", "\n\n  \n"); len(got) != 0 {
		t.Errorf("blank-only input should yield 0 chunks, got %d", len(got))
	}
}

// Regression: leading and consecutive blank lines must not drift line numbers.
// (The old `continue` skipped the currLine++, so StartLine/EndLine fell behind.)
func TestLineChunker_LeadingAndConsecutiveBlanks(t *testing.T) {
	// blank, blank, "alpha"(3), blank, blank, blank, "beta"(7)
	content := "\n\nalpha\n\n\n\nbeta\n"
	got := NewLineChunker().Chunk("x", content)

	if len(got) != 2 {
		t.Fatalf("want 2 chunks, got %d: %+v", len(got), got)
	}
	if got[0].StartLine != 3 || got[0].EndLine != 3 || got[0].Content != "alpha" {
		t.Errorf("chunk0 = %d-%d %q, want 3-3 'alpha'", got[0].StartLine, got[0].EndLine, got[0].Content)
	}
	if got[1].StartLine != 7 || got[1].EndLine != 7 || got[1].Content != "beta" {
		t.Errorf("chunk1 = %d-%d %q, want 7-7 'beta'", got[1].StartLine, got[1].EndLine, got[1].Content)
	}
}

// Content fidelity: indentation inside a chunk must be preserved, and a
// tab-only line counts as blank (chunk boundary).
func TestLineChunker_PreservesIndentationAndTabBlank(t *testing.T) {
	content := "func f() {\n\treturn 1\n}\n\t\n\tnext\n"
	got := NewLineChunker().Chunk("x", content)

	if len(got) != 2 {
		t.Fatalf("want 2 chunks (tab-only line splits), got %d: %+v", len(got), got)
	}
	if got[0].Content != "func f() {\n\treturn 1\n}" {
		t.Errorf("chunk0 content = %q, want indentation preserved", got[0].Content)
	}
	if got[1].Content != "\tnext" {
		t.Errorf("chunk1 content = %q, want '\\tnext'", got[1].Content)
	}
}
