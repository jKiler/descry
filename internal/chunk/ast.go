package chunk

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"

	"github.com/jKiler/descry/internal/core"
)

// ASTChunker splits Go source on real declaration boundaries (func / method /
// type) using the stdlib go/parser + go/ast, so each chunk is a semantic whole
// with exact line ranges — much better retrieval than blank-line splitting.
// For non-Go or unparseable files it delegates to Fallback (a LineChunker).
// It implements the Chunker interface, so it's a drop-in swap for LineChunker.
type ASTChunker struct {
	Fallback Chunker
}

// NewASTChunker returns an AST chunker backed by a LineChunker fallback.
func NewASTChunker() *ASTChunker {
	return &ASTChunker{Fallback: NewLineChunker()}
}

// ID identifies the chunking strategy for the reindex fingerprint.
func (c *ASTChunker) ID() string { return "ast-go" }

// Chunk parses a Go file into one chunk per top-level declaration (func, method,
// type, const, var), with the doc comment folded in and exact line ranges.
// Import blocks are skipped. Non-Go files, unparseable source, and files with no
// declarations all fall back to the LineChunker.
func (c *ASTChunker) Chunk(path, content string) []core.Chunk {
	chunks, _ := c.ChunkFile(path, content)
	return chunks
}

// ChunkFile is Chunk plus the name of the strategy that produced the chunks
// ("ast-go" or the fallback's id), so the evaluation harness can tell a working
// Go path from a file that quietly degraded to blank-line splitting.
func (c *ASTChunker) ChunkFile(path, content string) ([]core.Chunk, string) {
	_, isGoFile := strings.CutSuffix(path, ".go")
	if !isGoFile {
		return c.Fallback.Chunk(path, content), c.Fallback.ID()
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, content, parser.ParseComments)
	if err != nil {
		return c.Fallback.Chunk(path, content), c.Fallback.ID()
	}

	var chunks []core.Chunk
	seen := map[string]int{}
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.GenDecl:
			if d.Tok == token.IMPORT {
				continue
			}
			chunks = append(chunks, declChunk(path, content, fset, d.Doc, d, genDeclSymbol(d), seen))
		case *ast.FuncDecl:
			chunks = append(chunks, declChunk(path, content, fset, d.Doc, d, d.Name.Name, seen))
		}
	}

	if len(chunks) == 0 {
		return c.Fallback.Chunk(path, content), c.Fallback.ID()
	}

	return chunks, c.ID()
}

func declChunk(path, content string, fset *token.FileSet, doc *ast.CommentGroup, node ast.Node, symbol string, seen map[string]int) core.Chunk {
	start := node.Pos()
	if doc != nil {
		start = doc.Pos()
	}
	s, e := fset.Position(start), fset.Position(node.End())
	body := content[s.Offset:e.Offset]
	return core.Chunk{
		ID:        IDFor(path, body, seen),
		Path:      path,
		StartLine: s.Line,
		EndLine:   e.Line,
		Symbol:    symbol,
		Content:   body,
		StartByte: s.Offset,
		EndByte:   e.Offset,
	}
}

func genDeclSymbol(d *ast.GenDecl) string {
	if len(d.Specs) == 0 {
		return ""
	}
	switch s := d.Specs[0].(type) {
	case *ast.TypeSpec: // type Foo ...
		return s.Name.Name
	case *ast.ValueSpec: // const/var Foo ...
		if len(s.Names) > 0 {
			return s.Names[0].Name
		}
	case *ast.ImportSpec: // import block — no meaningful symbol
		return ""
	}
	return ""
}

var (
	_ Chunker    = (*ASTChunker)(nil)
	_ Diagnostic = (*ASTChunker)(nil)
)
