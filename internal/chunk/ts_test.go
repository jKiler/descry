package chunk

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/jKiler/descry/internal/core"
)

// coversEveryLine is the property that makes "never silently vanish from the
// index" checkable: the chunks of a parsed file, taken together, must include
// every line that carries meaning. A boundary bug that drops a declaration
// shows up here and nowhere else — recall metrics only notice if a query
// happens to target the lost lines.
//
// Lines with no word characters are exempt. A closing brace on its own line is
// not something anyone searches for, and emitting it as a chunk would be worse
// than leaving it out.
func coversEveryLine(t *testing.T, content string, chunks []core.Chunk) {
	t.Helper()
	lines := strings.Split(content, "\n")
	covered := make([]bool, len(lines)+2)
	for _, c := range chunks {
		for l := c.StartLine; l <= c.EndLine && l < len(covered); l++ {
			covered[l] = true
		}
	}
	for i, line := range lines {
		if !strings.ContainsFunc(line, isWordRune) {
			continue
		}
		if !covered[i+1] {
			t.Errorf("line %d not covered by any chunk: %q", i+1, line)
		}
	}
}

func isWordRune(r rune) bool {
	return r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')
}

func chunkSource(t *testing.T, path, src string) []core.Chunk {
	t.Helper()
	chunks, strategy := NewTSChunker().ChunkFile(path, src)
	if strategy == "line" {
		t.Fatalf("%s degraded to the fallback chunker", path)
	}
	coversEveryLine(t, src, chunks)
	return chunks
}

func symbols(chunks []core.Chunk) []string {
	var out []string
	for _, c := range chunks {
		if c.Symbol != "" {
			out = append(out, c.Symbol)
		}
	}
	return out
}

func hasSymbol(chunks []core.Chunk, want string) bool {
	for _, c := range chunks {
		if c.Symbol == want {
			return true
		}
	}
	return false
}

func findSymbol(t *testing.T, chunks []core.Chunk, want string) core.Chunk {
	t.Helper()
	for _, c := range chunks {
		if c.Symbol == want {
			return c
		}
	}
	t.Fatalf("no chunk with symbol %q; have %v", want, symbols(chunks))
	return core.Chunk{}
}

const pySrc = `"""Module doc.

Second paragraph, separated by a blank line that must not split it.
"""

import os


def top(a, b):
    """Add two things."""
    return a + b


class Store:
    """A store."""

    limit = 10

    @property
    def size(self):
        return self.limit

    def put(self, key, value):
        """Store a value."""
        self._data[key] = value
`

func TestPythonChunking(t *testing.T) {
	chunks := chunkSource(t, "pkg/store.py", pySrc)

	// Methods are qualified by their class; module-level functions are not.
	for _, want := range []string{"top", "Store.size", "Store.put"} {
		if !hasSymbol(chunks, want) {
			t.Errorf("missing symbol %q; have %v", want, symbols(chunks))
		}
	}

	// The module docstring spans a blank line. Splitting there would shred one
	// paragraph of documentation into fragments, so the blank line inside a
	// string literal must not be treated as a boundary.
	for _, c := range chunks {
		if strings.Contains(c.Content, "Module doc.") && !strings.Contains(c.Content, "Second paragraph") {
			t.Errorf("module docstring split at an unsafe blank line:\n%s", c.Content)
		}
	}

	// The decorator belongs to the method it decorates.
	if size := findSymbol(t, chunks, "Store.size"); !strings.Contains(size.Content, "@property") {
		t.Errorf("decorator not attached to its method:\n%s", size.Content)
	}

	// The doc comment is the most retrievable text a chunk has.
	if put := findSymbol(t, chunks, "Store.put"); !strings.Contains(put.Content, "Store a value.") {
		t.Errorf("docstring missing from method chunk:\n%s", put.Content)
	}
}

const goSrc = `package p

import "fmt"

// Add returns the sum.
func Add(a, b int) int { return a + b }

// T is a thing.
type T struct{ X int }

// M does something.
func (t T) M() string { return fmt.Sprint(t.X) }
`

func TestGoChunking(t *testing.T) {
	chunks := chunkSource(t, "pkg/p.go", goSrc)
	for _, want := range []string{"Add", "T", "T.M"} {
		if !hasSymbol(chunks, want) {
			t.Errorf("missing symbol %q; have %v", want, symbols(chunks))
		}
	}
	if add := findSymbol(t, chunks, "Add"); !strings.Contains(add.Content, "// Add returns the sum.") {
		t.Errorf("doc comment not attached:\n%s", add.Content)
	}
}

const rustSrc = `//! Crate docs.

use std::io;

/// A parser.
#[derive(Debug)]
pub struct Parser {
    pos: usize,
}

impl Parser {
    /// Build one.
    pub fn new() -> Self {
        Parser { pos: 0 }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn works() {
        assert!(true);
    }
}
`

func TestRustChunking(t *testing.T) {
	chunks, strategy := NewTSChunker().ChunkFile("src/lib.rs", rustSrc)
	if strategy == "line" {
		t.Fatal("rust degraded to the fallback chunker")
	}
	if !hasSymbol(chunks, "Parser.new") {
		t.Errorf("impl method not qualified by its type; have %v", symbols(chunks))
	}
	// The derive attribute is part of the struct's declaration.
	if p := findSymbol(t, chunks, "Parser"); !strings.Contains(p.Content, "#[derive(Debug)]") {
		t.Errorf("attribute not attached to its struct:\n%s", p.Content)
	}
	// An in-file #[cfg(test)] module is test code that file-level classification
	// cannot see; leaving it in makes a third of a Rust index tests.
	for _, c := range chunks {
		if strings.Contains(c.Content, "fn works") {
			t.Errorf("cfg(test) module was indexed:\n%s", c.Content)
		}
	}
}

const javaSrc = `package com.example;

/** A greeter. */
public class Greeter {

    private final String name;

    public Greeter(String name) {
        this.name = name;
    }

    /** Greets. */
    public String greet() {
        return "hi " + name;
    }
}
`

func TestJavaChunking(t *testing.T) {
	chunks := chunkSource(t, "src/main/java/com/example/Greeter.java", javaSrc)
	for _, want := range []string{"Greeter.Greeter", "Greeter.greet"} {
		if !hasSymbol(chunks, want) {
			t.Errorf("missing symbol %q; have %v", want, symbols(chunks))
		}
	}
	// The class's own javadoc and fields are not inside any method, and would
	// vanish without the cover's gap chunks.
	found := false
	for _, c := range chunks {
		if strings.Contains(c.Content, "A greeter.") && strings.Contains(c.Content, "private final String name") {
			found = true
		}
	}
	if !found {
		t.Error("class header (javadoc + fields) not kept as a chunk")
	}
}

const cSrc = `#include <stdio.h>

/* Adds. */
int add(int a, int b) {
    return a + b;
}

struct Point { int x; int y; };
`

func TestCChunking(t *testing.T) {
	chunks := chunkSource(t, "src/add.c", cSrc)
	for _, want := range []string{"add", "Point"} {
		if !hasSymbol(chunks, want) {
			t.Errorf("missing symbol %q; have %v", want, symbols(chunks))
		}
	}
}

const tsSrc = `import { x } from './x'

export interface Options {
  deep: boolean
}

export function reactive(target: object): object {
  return target
}

export const shallowRef = (value: unknown) => {
  return value
}

export class EffectScope {
  private active = true

  stop(): void {
    this.active = false
  }
}
`

func TestTypeScriptChunking(t *testing.T) {
	chunks := chunkSource(t, "packages/reactivity/src/index.ts", tsSrc)
	for _, want := range []string{"Options", "reactive", "shallowRef", "EffectScope.stop"} {
		if !hasSymbol(chunks, want) {
			t.Errorf("missing symbol %q; have %v", want, symbols(chunks))
		}
	}
	// `export` is part of the declaration a reader means, not a stray line above it.
	if r := findSymbol(t, chunks, "reactive"); !strings.HasPrefix(strings.TrimSpace(r.Content), "export function") {
		t.Errorf("export keyword not folded into the declaration:\n%s", r.Content)
	}
}

// TestOversizedNodeWindows checks the windowing mechanism: a function too long for the
// model's input becomes several overlapping windows, each cut at a statement
// boundary and each carrying the declaration's header in its embedded text —
// never a window holding only the signature.
func TestOversizedNodeWindows(t *testing.T) {
	var b strings.Builder
	b.WriteString("def huge(a):\n    \"\"\"Doc.\"\"\"\n")
	for i := 0; i < 200; i++ {
		b.WriteString("    result_value_number_")
		b.WriteString(strings.Repeat("x", 3))
		b.WriteString(" = compute_something(a, ")
		b.WriteString(strings.Repeat("y", 5))
		b.WriteString(")\n")
	}
	src := b.String()

	chunks := chunkSource(t, "big.py", src)
	if len(chunks) < 2 {
		t.Fatalf("a %d-line function was not split; got %d chunk(s)", strings.Count(src, "\n"), len(chunks))
	}
	for i, c := range chunks {
		if c.EmbedText == "" {
			t.Fatalf("window %d has no enriched embedded text", i)
		}
		if !strings.Contains(c.EmbedText, "def huge(a)") {
			t.Errorf("window %d does not repeat the declaration header:\n%s", i, first(c.EmbedText, 120))
		}
		// The raw content stays a clean slice of the file: the repeated header
		// belongs to the embedded text only.
		if strings.Contains(c.Content, "python big.py") {
			t.Errorf("window %d leaked the enriched header into raw content", i)
		}
		if estimateTokens(c.EmbedText) > 2*tsDefaultMaxTokens {
			t.Errorf("window %d is %d tokens, far past the %d budget", i, estimateTokens(c.EmbedText), tsDefaultMaxTokens)
		}
	}
	// No window may hold only the signature.
	if chunks[0].EndLine <= 2 {
		t.Errorf("first window ends at line %d — the signature was separated from the body", chunks[0].EndLine)
	}
}

func first(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// TestUnparseableFileDegrades pins the documented fallback: broken source never
// crashes an index and never vanishes from it.
func TestUnparseableFileDegrades(t *testing.T) {
	junk := "this is not python at all ((( \n\n\x00\x01 garbage\n"
	chunks, strategy := NewTSChunker().ChunkFile("broken.py", junk)
	if len(chunks) == 0 {
		t.Fatal("unparseable file produced no chunks — it vanished from the index")
	}
	t.Logf("strategy=%s chunks=%d", strategy, len(chunks))
}

// TestUnknownExtensionUsesFallback keeps prose and data files working: they have
// no grammar, and the paragraph chunker is the right unit for them.
func TestUnknownExtensionUsesFallback(t *testing.T) {
	chunks, strategy := NewTSChunker().ChunkFile("README.md", "# Title\n\nA paragraph.\n\nAnother.\n")
	if strategy != "line" {
		t.Errorf("markdown strategy = %q, want line", strategy)
	}
	if len(chunks) != 3 {
		t.Errorf("got %d chunks, want 3", len(chunks))
	}
}

// TestChunkIDsStableForUnchangedContent pins the incremental-indexing
// invariant: re-chunking identical content yields identical ids.
func TestChunkIDsStableForUnchangedContent(t *testing.T) {
	a, _ := NewTSChunker().ChunkFile("pkg/store.py", pySrc)
	b, _ := NewTSChunker().ChunkFile("pkg/store.py", pySrc)
	if len(a) != len(b) {
		t.Fatalf("chunk counts differ: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Content != b[i].Content || a[i].EmbedText != b[i].EmbedText {
			t.Errorf("chunk %d not stable across runs: %q vs %q", i, a[i].ID, b[i].ID)
		}
	}
}

// TestChunkIDsSurviveEditsElsewhere is the incremental-indexing invariant:
// inserting a line at the top of a file must not renumber every chunk below it.
// With line-keyed ids an unrelated import would give every chunk in the file a
// new identity, so an incremental index would re-insert them all and an
// evaluation diff could not tell a chunk that moved from one that changed.
func TestChunkIDsSurviveEditsElsewhere(t *testing.T) {
	before, _ := NewTSChunker().ChunkFile("pkg/store.py", pySrc)
	after, _ := NewTSChunker().ChunkFile("pkg/store.py", "import sys\n\n"+pySrc)

	ids := map[string]bool{}
	for _, c := range after {
		ids[c.ID] = true
	}
	moved := 0
	for _, c := range before {
		if strings.Contains(c.Content, "import os") {
			continue // the chunk the edit actually touched
		}
		if !ids[c.ID] {
			t.Errorf("chunk %q (%s) lost its id after an unrelated edit", c.Symbol, c.ID)
		}
		moved++
	}
	if moved == 0 {
		t.Fatal("no chunks compared")
	}
}

// TestDuplicateContentGetsDistinctIDs guards the store, which keys chunks by id
// with INSERT OR REPLACE: two identical bodies in one file must not collapse
// into one row.
func TestDuplicateContentGetsDistinctIDs(t *testing.T) {
	src := "def a():\n    pass\n\n\ndef b():\n    pass\n"
	chunks, _ := NewTSChunker().ChunkFile("dup.py", src)
	seen := map[string]bool{}
	for _, c := range chunks {
		if seen[c.ID] {
			t.Errorf("duplicate chunk id %q", c.ID)
		}
		seen[c.ID] = true
	}
}

// TestChunkersReportByteSpans keeps every chunker honest with the
// structural-integrity check, which compares byte offsets. A chunker that
// leaves them zero is not judged sound — it is not judged at all, and reports a
// perfect score it did not earn.
func TestChunkersReportByteSpans(t *testing.T) {
	cases := []struct {
		name string
		ck   Chunker
		path string
		src  string
	}{
		{"tree-sitter", NewTSChunker(), "pkg/store.py", pySrc},
		{"go/ast", NewASTChunker(), "pkg/p.go", goSrc},
		{"line", NewLineChunker(), "notes.txt", "one\ntwo\n\nthree\n\n\nfour\n"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, ch := range c.ck.Chunk(c.path, c.src) {
				if ch.EndByte <= ch.StartByte {
					t.Fatalf("chunk %q has no byte span (%d..%d)", ch.Symbol, ch.StartByte, ch.EndByte)
				}
				if ch.EndByte > len(c.src) {
					t.Fatalf("chunk %q ends past EOF (%d > %d)", ch.Symbol, ch.EndByte, len(c.src))
				}
				// The span must actually address the content that was stored.
				if got := strings.TrimSpace(c.src[ch.StartByte:ch.EndByte]); got != strings.TrimSpace(ch.Content) {
					t.Errorf("chunk %q byte span does not match its content:\n span: %q\n content: %q",
						ch.Symbol, got, ch.Content)
				}
			}
		})
	}
}

// Every setting that can change what a chunk contains must change the id, and
// only non-defaults may appear in it. The id is what keeps a configured
// chunker's output out of the default's index and embed cache; a setting
// missing from it means two differently-chunked indexes share a fingerprint and
// silently skip the rebuild that would have corrected them.
func TestChunkerIDCoversEverySettingThatChangesChunks(t *testing.T) {
	if got := NewTSChunker().ID(); got != "ts" {
		t.Fatalf("default id is %q, want %q", got, "ts")
	}
	for _, tc := range []struct {
		name string
		set  func(*TSChunker)
	}{
		{"bodysig", func(c *TSChunker) { c.SignatureFromBody = true }},
		{"cap", func(c *TSChunker) { c.CapHeader = true }},
		{"both", func(c *TSChunker) { c.SignatureFromBody, c.CapHeader = true, true }},
		{"maxtokens", func(c *TSChunker) { c.MaxTokens = 80 }},
		{"overlap", func(c *TSChunker) { c.OverlapStatements = 3 }},
		{"mingap", func(c *TSChunker) { c.MinGapLines = 4 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v := NewTSChunker()
			tc.set(v)
			if v.ID() == "ts" {
				t.Errorf("id is still %q; this chunker's output would reuse the "+
					"default's index and cached vectors", v.ID())
			}
		})
	}
}

// And the settings must genuinely change the output, or the option is a no-op
// that only fragments indexes. Demonstrated on a deep path: the header cap only
// binds when the header is fat, which is the case it exists for.
func TestChunkerOptionsChangeTheEmbeddedText(t *testing.T) {
	shipped := NewTSChunker()
	java := "spring-boot-project/spring-boot-autoconfigure/src/main/java/" +
		"org/springframework/boot/autoconfigure/web/servlet/error/Greeter.java"
	for _, tc := range []struct {
		name string
		set  func(*TSChunker)
	}{
		{"bodysig", func(c *TSChunker) { c.SignatureFromBody = true }},
		{"cap", func(c *TSChunker) { c.CapHeader = true }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v := NewTSChunker()
			tc.set(v)
			if sameEmbedText(shipped.Chunk(java, javaSrc), v.Chunk(java, javaSrc)) {
				t.Error("produces the same embedded text as the default")
			}
		})
	}
}

func sameEmbedText(a, b []core.Chunk) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].EmbedText != b[i].EmbedText {
			return false
		}
	}
	return true
}

// The signature is bounded by a *byte* budget, so the cut can land inside a
// multi-byte character. It must not: this string is concatenated into the text
// handed to the tokenizer, and half a rune there is malformed UTF-8 in the
// embedded representation of every chunk of that declaration.
func TestSignatureTruncationKeepsRunesIntact(t *testing.T) {
	// Three-byte runes, because sigMaxBytes is not a multiple of three — a
	// two-byte script would land on a boundary by luck and prove nothing.
	src := []byte(strings.Repeat("語", 100))
	got := signatureText(src, 0, uint(len(src)))
	if !utf8.ValidString(got) {
		t.Errorf("truncated signature is not valid UTF-8: %q", got)
	}
	if len(got) > sigMaxBytes {
		t.Errorf("signature is %d bytes, over the %d budget", len(got), sigMaxBytes)
	}
}

// The cover must not depend on the alphabet. worthIndexing decides whether an
// uncovered region survives, and it runs before the line-count check, so an
// alphabet-specific test here silently drops content from the index.
func TestWorthIndexingIsNotASCIIOnly(t *testing.T) {
	for _, s := range []string{
		"这是一个中文注释",       // Chinese
		"// комментарий", // Cyrillic
		"σταθερά = 1",    // Greek
	} {
		if !worthIndexing(s, 1) {
			t.Errorf("%q judged not worth indexing; it would vanish from the cover", s)
		}
	}
	// Still noise, in any alphabet.
	if worthIndexing("  }\n)\n", 1) {
		t.Error("punctuation-only region judged worth indexing")
	}
}
