package graph

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/packages"
)

// BuildFromGoDirTyped builds the same kind of call graph as BuildFromGoDir, but
// resolves every callee with the type checker (go/types) instead of matching by
// name. Two methods named Do on different types become DISTINCT nodes, and a
// call like x.Method() resolves to the exact method on x's real type — things
// the name-based heuristic can't do. Node ids come from nodeID (below).
//
// Requires golang.org/x/tools/go/packages, and `root` must be a module (contain
// a go.mod) because packages.Load shells out to the go toolchain.
//
// Interface/dynamic dispatch still resolves only to the interface method — that
// needs go/ssa + callgraph/{cha,rta,vta}, which is out of scope.
func BuildFromGoDirTyped(root string) (*Graph, error) {
	g := New()

	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedSyntax | packages.NeedTypes |
			packages.NeedTypesInfo | packages.NeedDeps | packages.NeedImports,
		Dir: root,
	}
	pkgs, err := packages.Load(cfg, "./...")
	if err != nil {
		return nil, err
	}

	for _, pkg := range pkgs {
		info := pkg.TypesInfo
		if info == nil {
			continue // package failed to type-check; skip it (best-effort)
		}
		for _, file := range pkg.Syntax {
			for _, decl := range file.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok || fn.Body == nil {
					continue
				}
				// info.Defs maps the declaring identifier to its object; that's
				// the *types.Func for this function/method — a unique caller id.
				callerObj := info.Defs[fn.Name]
				if callerObj == nil {
					continue
				}
				caller := nodeID(callerObj)

				ast.Inspect(fn.Body, func(n ast.Node) bool {
					call, ok := n.(*ast.CallExpr)
					if !ok {
						return true
					}
					if callee := resolveCallee(info, call.Fun); callee != nil {
						g.AddEdge(caller, nodeID(callee))
					}
					return true
				})
			}
		}
	}
	return g, nil
}

// resolveCallee resolves a call's function expression to the *types.Func it
// denotes, or nil if it isn't a statically-known function/method (e.g. calling
// the result of another call, or a variable of func type).
func resolveCallee(info *types.Info, fun ast.Expr) types.Object {
	switch f := fun.(type) {
	case *ast.Ident: // foo()
		return asFunc(info.Uses[f])
	case *ast.SelectorExpr: // x.Method() or pkg.Func()
		if sel, ok := info.Selections[f]; ok {
			return asFunc(sel.Obj()) // method/field selected on a value
		}
		return asFunc(info.Uses[f.Sel]) // package-qualified function (pkg isn't a value)
	}
	return nil
}

// asFunc returns obj if it is a *types.Func, else nil. A nil obj is fine — the
// type assertion just reports ok == false.
func asFunc(obj types.Object) types.Object {
	if _, ok := obj.(*types.Func); ok {
		return obj
	}
	return nil
}

// nodeID returns a stable, collision-free id for a resolved object:
//
//	function -> "m/pkg.Func"
//	method   -> "(m/pkg.Type).Method"   (via *types.Func.FullName)
//
// The receiver type in a method's id is what keeps BM25.Search and
// Hybrid.Search from collapsing into one node.
func nodeID(obj types.Object) string {
	if fn, ok := obj.(*types.Func); ok {
		return fn.FullName()
	}
	if obj.Pkg() != nil {
		return obj.Pkg().Path() + "." + obj.Name()
	}
	return obj.Name()
}
