package chunk

import (
	"strings"
	"testing"
)

// A tiny Go file: a documented type, a documented func, and an undocumented
// method. Line numbers are 1-based from the backtick.
//
//	1  package sample
//	2
//	3  // Greeter greets people.
//	4  type Greeter struct {
//	5      Name string
//	6  }
//	7
//	8  // Hello returns a greeting.
//	9  func Hello(name string) string {
//	10     return "hi " + name
//	11 }
//	12
//	13 func (g Greeter) Greet() string {
//	14     return "hi " + g.Name
//	15 }
const astSample = `package sample

// Greeter greets people.
type Greeter struct {
	Name string
}

// Hello returns a greeting.
func Hello(name string) string {
	return "hi " + name
}

func (g Greeter) Greet() string {
	return "hi " + g.Name
}
`

// AST chunking: one chunk per top-level declaration, doc comments included,
// with correct symbol names and line ranges.
func TestASTChunker_SplitsOnDeclarations(t *testing.T) {
	got := NewASTChunker().Chunk("sample.go", astSample)
	if len(got) != 3 {
		t.Fatalf("want 3 chunks (type + 2 funcs), got %d: %+v", len(got), got)
	}

	want := []struct {
		symbol             string
		startLine, endLine int
		contains           string
	}{
		{"Greeter", 3, 6, "type Greeter struct"},           // starts at the doc comment
		{"Hello", 8, 11, "func Hello(name string) string"}, // starts at the doc comment
		{"Greet", 13, 15, "func (g Greeter) Greet()"},      // no doc, starts at func
	}
	for i, w := range want {
		if got[i].Symbol != w.symbol {
			t.Errorf("chunk%d symbol = %q, want %q", i, got[i].Symbol, w.symbol)
		}
		if got[i].StartLine != w.startLine || got[i].EndLine != w.endLine {
			t.Errorf("chunk%d lines = %d-%d, want %d-%d", i, got[i].StartLine, got[i].EndLine, w.startLine, w.endLine)
		}
		if !strings.Contains(got[i].Content, w.contains) {
			t.Errorf("chunk%d content missing %q; got %q", i, w.contains, got[i].Content)
		}
	}
	if got[0].Path != "sample.go" {
		t.Errorf("path = %q, want sample.go", got[0].Path)
	}
	if got[1].ID != "sample.go#8" {
		t.Errorf("chunk1 id = %q, want sample.go#8", got[1].ID)
	}
}

// Non-Go files fall back to the line chunker.
func TestASTChunker_FallsBackForNonGo(t *testing.T) {
	content := "para one\n\npara two\n"
	got := NewASTChunker().Chunk("notes.md", content)
	if len(got) != 2 {
		t.Fatalf("non-.go should fall back to line chunker (2 paragraphs), got %d: %+v", len(got), got)
	}
}

// Unparseable Go falls back too — must not panic, must not return zero chunks.
func TestASTChunker_FallsBackOnParseError(t *testing.T) {
	got := NewASTChunker().Chunk("broken.go", "this is not valid go {{{\n")
	if len(got) == 0 {
		t.Errorf("parse error should fall back to line chunker, got 0 chunks")
	}
}
