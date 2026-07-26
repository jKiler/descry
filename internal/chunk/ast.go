package chunk

import (
	"fmt"
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
	_, isGoFile := strings.CutSuffix(path, ".go")
	if !isGoFile {
		return c.Fallback.Chunk(path, content)
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, content, parser.ParseComments)
	if err != nil {
		return c.Fallback.Chunk(path, content)
	}

	var chunks []core.Chunk
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.GenDecl:
			if d.Tok == token.IMPORT {
				continue
			}
			chunks = append(chunks, declChunk(path, content, fset, d.Doc, d, genDeclSymbol(d)))
		case *ast.FuncDecl:
			chunks = append(chunks, declChunk(path, content, fset, d.Doc, d, d.Name.Name))
		}
	}

	if len(chunks) == 0 {
		return c.Fallback.Chunk(path, content)
	}

	return chunks
}

func declChunk(path, content string, fset *token.FileSet, doc *ast.CommentGroup, node ast.Node, symbol string) core.Chunk {
	start := node.Pos()
	if doc != nil {
		start = doc.Pos()
	}
	s, e := fset.Position(start), fset.Position(node.End())
	return core.Chunk{
		ID:        fmt.Sprintf("%s#%d", path, s.Line),
		Path:      path,
		StartLine: s.Line,
		EndLine:   e.Line,
		Symbol:    symbol,
		Content:   content[s.Offset:e.Offset],
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

var _ Chunker = (*ASTChunker)(nil)
