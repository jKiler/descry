// Package lang is the language pack registry: one declarative record per
// supported language, holding everything the rest of descry needs to know about
// it — which files belong to it, which of those are test code, and (once a
// grammar is attached) how to cut it into syntactic chunks.
//
// The point of the registry is that adding a language is adding a Pack, never
// editing the chunker, the indexer, or anything else. If supporting a new
// language requires a change outside its own pack file, that is a bug in this
// interface, not in the language.
package lang

import (
	"path"
	"strings"
	"sync"

	ts "github.com/tree-sitter/go-tree-sitter"
)

// Pack describes one language. Everything is data: the core code branches on
// the presence of a field, never on the language's name.
type Pack struct {
	// Name is the pack's identity — it appears in chunk metadata and in the
	// enriched embedded text, so keep it the word a person would use ("python",
	// "typescript", "cpp").
	Name string

	// Exts are the lowercase file extensions this pack claims, leading dot
	// included. Two packs must not claim the same extension.
	Exts []string

	// TestBases are globs (path.Match syntax) matched against a file's base
	// name to recognize test code, TestDirs are directory names that mark a
	// whole subtree as test code, and TestPaths are slash-joined path fragments
	// that do the same for languages whose convention spans two segments
	// ("src/test/java"). Test code bloats the index and crowds out the
	// production code people actually search for; the Go-only `_test.go` rule
	// this generalizes has been in the indexer from the start.
	TestBases []string
	TestDirs  []string
	TestPaths []string

	// GeneratedMarkers are substrings that, appearing in a file's first few
	// lines, mark it as machine-generated.
	GeneratedMarkers []string

	// Grammar, when set, returns the tree-sitter language for this pack. Nil
	// means the pack contributes file classification only, and its files chunk
	// through the fallback chunker.
	Grammar func() *ts.Language

	// Query is the tree-sitter query (S-expression) naming the chunkable units.
	// Its captures are the whole interface to the chunker:
	//
	//	@chunk      this node becomes a chunk of its own
	//	@scope      a container: qualifies the names inside it, is not a chunk
	//	@drop       excluded from the index entirely (in-file test modules)
	//	@name       the symbol name of the @chunk or @scope it is captured with
	//	@qualifier  an extra prefix for the symbol (Go's method receiver type)
	//	@body       the declaration's body; everything before it is the signature,
	//	            which is what each window of an oversized chunk repeats
	//
	// A language that cannot express a boundary in this vocabulary is a gap in
	// the vocabulary, not grounds for special-casing the language in the core.
	Query string

	// AttachKinds are node kinds that belong to the declaration below them and
	// should be folded into its chunk, beyond comments (which are recognized
	// structurally). Rust attribute items are the motivating case: `#[derive(…)]`
	// above a struct is part of that struct's declaration.
	AttachKinds []string

	// WrapKinds are node kinds that merely decorate the declaration they end
	// with — `export …` in TypeScript, `template <…> …` in C++. A captured node
	// that is the last child of such a wrapper extends up to the wrapper's
	// start, so the chunk begins where a reader would say the declaration
	// begins instead of stranding the keyword in a one-line gap chunk.
	WrapKinds []string

	// Disambiguate resolves an extension two languages share, by naming the
	// pack that should actually handle this file's content. Returning "" keeps
	// this pack.
	//
	// `.h` is the case that forces this to exist: the extension says nothing
	// about whether the file is C or C++, and a C++ header parsed by the C
	// grammar does not fail loudly — it produces a plausible-looking but wrong
	// tree, in which a class's methods lose their class and a template's body
	// is one enormous malformed declaration.
	Disambiguate func(content string) string
}

var (
	mu     sync.RWMutex
	packs  []*Pack
	byExt  = map[string]*Pack{}
	byName = map[string]*Pack{}
)

// Register adds a pack. It panics on a duplicate extension: two packs claiming
// the same files is a configuration error that must fail loudly at init, not
// silently pick a winner at index time.
func Register(p *Pack) {
	mu.Lock()
	defer mu.Unlock()
	for _, e := range p.Exts {
		if other, dup := byExt[e]; dup {
			panic("lang: extension " + e + " claimed by both " + other.Name + " and " + p.Name)
		}
		byExt[e] = p
	}
	byName[p.Name] = p
	packs = append(packs, p)
}

// ForPath returns the pack owning a file, or nil if no pack claims it.
func ForPath(p string) *Pack {
	mu.RLock()
	defer mu.RUnlock()
	return byExt[strings.ToLower(path.Ext(p))]
}

// ByName returns a pack by name, or nil.
func ByName(n string) *Pack {
	mu.RLock()
	defer mu.RUnlock()
	return byName[n]
}

// All returns every registered pack, in registration order.
func All() []*Pack {
	mu.RLock()
	defer mu.RUnlock()
	out := make([]*Pack, len(packs))
	copy(out, packs)
	return out
}

// Extensions returns every extension any pack claims.
func Extensions() []string {
	mu.RLock()
	defer mu.RUnlock()
	out := make([]string, 0, len(byExt))
	for e := range byExt {
		out = append(out, e)
	}
	return out
}

// IsTest reports whether rel (a slash-separated path relative to the index
// root) is test code, per its pack's declared conventions.
func IsTest(rel string) bool {
	p := ForPath(rel)
	if p == nil {
		return false
	}
	rel = strings.ReplaceAll(rel, "\\", "/")
	base := path.Base(rel)
	for _, g := range p.TestBases {
		if ok, _ := path.Match(g, base); ok {
			return true
		}
	}
	segs := strings.Split(path.Dir(rel), "/")
	for _, s := range segs {
		for _, d := range p.TestDirs {
			if s == d {
				return true
			}
		}
	}
	for _, frag := range p.TestPaths {
		if strings.Contains("/"+rel, "/"+frag+"/") {
			return true
		}
	}
	return false
}

// IsGenerated reports whether head (the first few lines of a file) carries a
// generated-code marker for the file's language.
func IsGenerated(rel, head string) bool {
	p := ForPath(rel)
	if p == nil {
		return false
	}
	for _, m := range p.GeneratedMarkers {
		if strings.Contains(head, m) {
			return true
		}
	}
	return false
}

// ForFile is ForPath refined by the file's content, for extensions two
// languages share. Chunking should use this; deciding whether to index a file
// at all can use ForPath, since both candidates agree the file belongs.
func ForFile(p, content string) *Pack {
	pack := ForPath(p)
	if pack == nil || pack.Disambiguate == nil {
		return pack
	}
	if name := pack.Disambiguate(content); name != "" {
		if other := ByName(name); other != nil {
			return other
		}
	}
	return pack
}
