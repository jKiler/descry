package graph

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strings"
)

// BuildFromGoDir walks root, parses every .go file, and builds a symbol-level
// call graph: an edge caller -> callee for each call expression inside a
// function body. Resolution is name-based, so same-named functions collide and
// interface/dynamic dispatch is missed — fast and module-independent, works on
// code that doesn't compile. For exact resolution, see BuildFromGoDirTyped.
// Files that fail to parse (or aren't Go) are skipped, so it never aborts.
func BuildFromGoDir(root string) (*Graph, error) {
	g := New()
	fset := token.NewFileSet()

	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Skip hidden and vendored dirs, but never the root itself.
			if path != root && (strings.HasPrefix(d.Name(), ".") || d.Name() == "vendor") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}

		// nil src makes ParseFile read the file from disk.
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return nil // skip unparseable files; don't abort the walk
		}

		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			caller := fn.Name.Name
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				if call, ok := n.(*ast.CallExpr); ok {
					if callee := calleeName(call.Fun); callee != "" {
						g.AddEdge(caller, callee)
					}
				}
				return true // keep descending into the subtree
			})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return g, nil
}

// calleeName extracts the called identifier from a call's function expression:
//
//	foo()      -> "foo"     (*ast.Ident)
//	pkg.Foo()  -> "Foo"     (*ast.SelectorExpr — take .Sel.Name)
//	x.Method() -> "Method"
//
// Anything else (e.g. calling the result of another call) -> "".
func calleeName(fun ast.Expr) string {
	switch f := fun.(type) {
	case *ast.Ident:
		return f.Name
	case *ast.SelectorExpr:
		return f.Sel.Name
	}
	return ""
}
