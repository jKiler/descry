package chunk

import (
	"fmt"
	"sort"
	"strings"

	"github.com/jKiler/descry/internal/core"
	"github.com/jKiler/descry/internal/lang"
	ts "github.com/tree-sitter/go-tree-sitter"
)

// TSChunker cuts source on real syntactic boundaries using tree-sitter.
//
// The whole language-specific story lives in the pack (internal/lang): its
// grammar and its query. This file knows only the capture vocabulary — @chunk,
// @scope, @drop, @name, @qualifier, @body — and never branches on a
// language's name. Adding a language is adding a pack; if it ever requires
// editing this file, the vocabulary is what needs fixing, not the language.
//
// The cut is a *cover*: every byte of a parsed file lands in exactly one chunk.
// Captured declarations become chunks, and whatever they leave uncovered
// (imports, module-level constants, fields declared after the methods) becomes
// a gap chunk attributed to its enclosing scope. Nothing in a parsed file can
// silently fail to be indexed.
type TSChunker struct {
	// Fallback chunks files with no grammar, and files the grammar cannot make
	// sense of. Never nil in a chunker built by NewTSChunker.
	Fallback Chunker

	// MaxTokens is the embedded-text budget in wordpiece tokens. A chunk over
	// budget is split into overlapping windows cut on statement boundaries, so
	// a long function's tail gets embedded instead of being truncated away by
	// the model's 256-token input limit.
	MaxTokens int

	// OverlapStatements is how many trailing statements of a window repeat at
	// the head of the next one, so a fact spanning a window boundary survives
	// intact in at least one window.
	OverlapStatements int

	// MinGapLines drops uncovered regions shorter than this that carry no
	// identifier — stray braces and blank lines, which are noise as chunks.
	MinGapLines int

	// CapHeader bounds the enriched header to a fraction of MaxTokens.
	//
	// Off by default, on measurement rather than principle. Capping it looked
	// obviously right — it was aimed at Java, where the header ate a third of
	// the budget — but it cost Go 6.7pp of chunk Recall@10 and measured worse
	// across most corpora.
	CapHeader bool

	// SignatureFromBody ends the signature where the declaration's body begins,
	// which is what the @body capture means.
	//
	// Also off by default, and this one is genuinely uncomfortable: with it off
	// the "signature" in a chunk's embedded text is the declaration collapsed
	// onto one line and truncated — a *preview of the code* rather than a
	// signature. That arrived as a bug (no pack emits @signature, which is all
	// the chunker used to look for), and fixing it made six of seven corpora
	// worse or flat, Go worst of all.
	//
	// Shipping the empirically better behaviour over the obviously correct one
	// is deliberate, but the mechanism is not understood — which is why this is
	// a field rather than a fix.
	SignatureFromBody bool

	pools *parserPools
}

// NewTSChunker returns a tree-sitter chunker with the shipped defaults and a
// LineChunker fallback.
func NewTSChunker() *TSChunker {
	return &TSChunker{
		Fallback:          NewLineChunker(),
		MaxTokens:         tsDefaultMaxTokens,
		OverlapStatements: 1,
		MinGapLines:       1,
		CapHeader:         false,
		SignatureFromBody: false,
		pools:             newParserPools(),
	}
}

// tsDefaultMaxTokens is the per-chunk budget. The MiniLM export accepts 256
// wordpiece tokens including [CLS]/[SEP], and estimateTokens carries a measured
// p05/p95 spread of 0.82/1.19 around the true count, so the budget sits well
// under the ceiling: at the pessimistic end of that spread a 220-token estimate
// is a ~269-token reality, which loses only a few tokens, while a budget at the
// ceiling would silently truncate a long function's tail — invisible in the
// index, and visible only as a coverage miss.
const tsDefaultMaxTokens = 220

// ID identifies the chunking strategy for the reindex fingerprint. It is derived
// from the settings that change what a chunk contains, rather than set by hand,
// so a caller who constructs a TSChunker with non-default options cannot have
// its chunks silently reuse the default's index or embed-cache entries.
func (c *TSChunker) ID() string {
	id := "ts"
	if c.SignatureFromBody {
		id += "+bodysig"
	}
	if c.CapHeader {
		id += "+cap"
	}
	// The numeric settings move chunk boundaries, so they belong in the id for
	// the same reason the booleans do. Only non-defaults appear, so the shipped
	// chunker keeps the bare "ts" and no user's index is invalidated by this.
	if c.MaxTokens != tsDefaultMaxTokens {
		id += fmt.Sprintf("+max%d", c.MaxTokens)
	}
	if c.OverlapStatements != 1 {
		id += fmt.Sprintf("+ov%d", c.OverlapStatements)
	}
	if c.MinGapLines != 1 {
		id += fmt.Sprintf("+gap%d", c.MinGapLines)
	}
	return id
}

func init() { register("ts", func() Chunker { return NewTSChunker() }) }

// Chunk implements Chunker.
func (c *TSChunker) Chunk(path, content string) []core.Chunk {
	chunks, _ := c.ChunkFile(path, content)
	return chunks
}

// ChunkFile is Chunk plus the strategy that produced the chunks: the pack name
// when the grammar handled the file, or the fallback's id when it degraded.
//
// Degradation is deliberate and total: a file with no pack, no grammar, a
// parser failure, or a parse yielding no captured declaration at all goes to
// the fallback whole. It is never dropped, and a malformed file never fails an
// index.
func (c *TSChunker) ChunkFile(path, content string) ([]core.Chunk, string) {
	p := lang.ForFile(path, content)
	if p == nil || p.Grammar == nil || p.Query == "" {
		return c.Fallback.Chunk(path, content), c.Fallback.ID()
	}
	compiled, err := compiledFor(p)
	if err != nil {
		return c.Fallback.Chunk(path, content), c.Fallback.ID()
	}
	src := []byte(content)

	tree := c.pools.parse(p, compiled.language, src)
	if tree == nil {
		return c.Fallback.Chunk(path, content), c.Fallback.ID()
	}
	defer tree.Close()

	units := collect(compiled, tree.RootNode(), src, p, c.SignatureFromBody)
	if !hasChunk(units) {
		return c.Fallback.Chunk(path, content), c.Fallback.ID()
	}

	chunks := c.assemble(path, src, p, units, tree.RootNode())
	if len(chunks) == 0 {
		return c.Fallback.Chunk(path, content), c.Fallback.ID()
	}
	return chunks, p.Name
}

// unit is one captured region of a file, in bytes.
type unit struct {
	start, end uint
	name       string // symbol name from @name, "" when unnamed
	scope      bool   // a container: qualifies nested names, is not a chunk itself
	drop       bool   // excluded from the index, and from the cover
	qualifier  string // an extra name prefix (a Go receiver type), "" if none
	sigEnd     uint   // end of the declaration header; == start when unknown
	node       *ts.Node
}

// hasChunk reports whether the query found anything to make a chunk out of.
// Scopes do not count (a class with no members is a gap chunk, not a unit), and
// neither do drops — a Rust file whose only match is a `#[cfg(test)] mod` has
// found nothing to index, and belongs to the fallback.
func hasChunk(us []unit) bool {
	for _, u := range us {
		if !u.scope && !u.drop {
			return true
		}
	}
	return false
}

// collect runs the pack's query and turns the captures into units, dropping
// anything inside a @drop region and any chunk another chunk already contains.
func collect(q *compiledQuery, root *ts.Node, src []byte, p *lang.Pack, sigFromBody bool) []unit {
	cursor := ts.NewQueryCursor()
	defer cursor.Close()
	matches := cursor.Matches(q.query, root, src)

	var chunks, scopes, drops []unit
	for m := matches.Next(); m != nil; m = matches.Next() {
		var node, nameNode, bodyNode, qualNode *ts.Node
		kind := ""
		for i := range m.Captures {
			// QueryMatch.Captures aliases memory the cursor reuses on the next
			// match, so every node that outlives this iteration must be copied out
			// by value. A ts.Node is self-contained (it references the tree, not
			// the match), so the copy stays valid for the tree's lifetime.
			n := m.Captures[i].Node
			switch q.name(m.Captures[i].Index) {
			case capChunk:
				node, kind = &n, capChunk
			case capScope:
				node, kind = &n, capScope
			case capDrop:
				node, kind = &n, capDrop
			case capName:
				nameNode = &n
			case capQualifier:
				qualNode = &n
			case capBody:
				bodyNode = &n
			}
		}
		if node == nil {
			continue
		}
		u := unit{start: node.StartByte(), end: node.EndByte(),
			scope: kind == capScope, drop: kind == capDrop, node: node}
		if nameNode != nil {
			u.name = string(src[nameNode.StartByte():nameNode.EndByte()])
		}
		if qualNode != nil {
			u.qualifier = string(src[qualNode.StartByte():qualNode.EndByte()])
		}
		// `export`/`template` wrappers are part of the declaration, and so are the
		// comments and attributes immediately above it: the doc comment is usually
		// the most retrievable text the chunk has. Dropped regions extend the same
		// way, so a dropped test module takes its `#[cfg(test)]` with it instead of
		// leaving the attribute behind as a stray chunk.
		outer := unwrap(node, p)
		u.start = min(outer.StartByte(), attachLeading(outer, src, p))
		u.end = max(u.end, outer.EndByte())
		// The signature runs from the declaration's start to the start of its
		// body, so @body is what bounds it. A declaration with no body — an
		// abstract method, a type alias — is all signature.
		u.sigEnd = u.end
		if sigFromBody && bodyNode != nil && bodyNode.StartByte() > u.start {
			u.sigEnd = bodyNode.StartByte()
		}
		switch kind {
		case capChunk:
			chunks = append(chunks, u)
		case capScope:
			scopes = append(scopes, u)
		case capDrop:
			drops = append(drops, u)
		}
	}

	chunks = removeInside(chunks, drops)
	scopes = removeInside(scopes, drops)
	drops = dedupe(drops)
	// Two chunk captures nest when a grammar offers both a wrapper and its
	// payload (a decorated definition and the definition inside it, an exported
	// declaration and the declaration). The outer one is the whole declaration a
	// reader means, so it wins.
	chunks = keepOutermost(dedupe(chunks))
	out := append(chunks, dedupe(scopes)...)
	return append(out, drops...)
}

// unwrap climbs past wrapper nodes that decorate the declaration they end with
// (`export class …`, `template <T> void f() …`), so the chunk starts where a
// reader would say the declaration starts. Only the wrapper's *last* named
// child qualifies, which is what distinguishes a decorating wrapper from a
// container that merely happens to hold the node.
func unwrap(node *ts.Node, p *lang.Pack) *ts.Node {
	for len(p.WrapKinds) > 0 {
		parent := node.Parent()
		if parent == nil || !contains(p.WrapKinds, parent.Kind()) {
			return node
		}
		if last := parent.NamedChild(parent.NamedChildCount() - 1); last == nil || last.Id() != node.Id() {
			return node
		}
		node = parent
	}
	return node
}

// attachLeading walks backwards over the siblings immediately above node,
// absorbing comment and attribute nodes separated from it by no blank line.
// Comment *kind* is recognized structurally (any node kind containing
// "comment"), which holds across every grammar descry ships, so the rule needs
// no per-language configuration.
func attachLeading(node *ts.Node, src []byte, p *lang.Pack) uint {
	start := node.StartByte()
	cur := node
	for {
		prev := cur.PrevNamedSibling()
		if prev == nil {
			return start
		}
		kind := prev.Kind()
		if !strings.Contains(kind, "comment") && !contains(p.AttachKinds, kind) {
			return start
		}
		if blankLineBetween(src, prev.EndByte(), start) {
			return start // a blank line separates a floating comment from the decl
		}
		start = prev.StartByte()
		cur = prev
	}
}

func blankLineBetween(src []byte, from, to uint) bool {
	if from > to || to > uint(len(src)) {
		return false
	}
	// A blank line is "\n\n" on unix and "\r\n\r\n" on windows; testing only
	// for the former attaches a comment across a genuine blank line in every
	// CRLF file. Drop the carriage returns rather than matching both spellings,
	// which also covers a mixed-ending file.
	between := strings.ReplaceAll(string(src[from:to]), "\r", "")
	return strings.Contains(between, "\n\n")
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// removeInside drops units contained in any of the given regions.
func removeInside(us, regions []unit) []unit {
	if len(regions) == 0 {
		return us
	}
	out := us[:0]
	for _, u := range us {
		inside := false
		for _, r := range regions {
			if u.start >= r.start && u.end <= r.end {
				inside = true
				break
			}
		}
		if !inside {
			out = append(out, u)
		}
	}
	return out
}

// dedupe removes units with identical spans, which several query patterns
// matching the same node produce, preferring the variant that carries a name.
func dedupe(us []unit) []unit {
	sortUnits(us)
	var out []unit
	for _, u := range us {
		if n := len(out); n > 0 && u.start == out[n-1].start && u.end == out[n-1].end {
			if out[n-1].name == "" && u.name != "" {
				out[n-1] = u
			}
			continue
		}
		out = append(out, u)
	}
	return out
}

// keepOutermost drops any chunk unit fully contained in another chunk unit.
func keepOutermost(us []unit) []unit {
	sortUnits(us)
	var out []unit
	var lastEnd uint
	for _, u := range us {
		if len(out) > 0 && u.end <= lastEnd {
			continue
		}
		out = append(out, u)
		lastEnd = u.end
	}
	return out
}

func sortUnits(us []unit) {
	sort.Slice(us, func(i, j int) bool {
		if us[i].start != us[j].start {
			return us[i].start < us[j].start
		}
		return us[i].end > us[j].end
	})
}

// compiledQuery is a pack's query compiled once, with capture indices resolved.
type compiledQuery struct {
	query    *ts.Query
	language *ts.Language
	names    []string
}

func (q *compiledQuery) name(i uint32) string {
	if int(i) < len(q.names) {
		return q.names[i]
	}
	return ""
}

// The capture vocabulary. This is the entire contract between a language pack
// and the chunker.
const (
	capChunk     = "chunk"     // becomes a chunk of its own
	capScope     = "scope"     // qualifies names inside it; yields no chunk itself
	capDrop      = "drop"      // excluded from the index entirely
	capName      = "name"      // the symbol name of the captured @chunk/@scope
	capQualifier = "qualifier" // an extra prefix for the symbol (a Go receiver type)
	capBody      = "body"      // the declaration's body; its start ends the signature
	capContext   = "context"   // file-level context (package clause, module header)
)

func newCompiled(p *lang.Pack) (*compiledQuery, error) {
	l := p.Grammar()
	if l == nil {
		// A pack whose Grammar field is set but whose function returns nil. Not
		// reachable from the packs in this repository, and guarded anyway: the
		// premise of internal/lang is that a language is added by writing a pack
		// and nothing else, so a malformed pack has to degrade its own language
		// to the fallback the way every other bad input here does, not panic the
		// indexer inside the binding.
		return nil, fmt.Errorf("lang %s: grammar is nil", p.Name)
	}
	q, qerr := ts.NewQuery(l, p.Query)
	if qerr != nil {
		return nil, fmt.Errorf("lang %s: query: %s", p.Name, qerr.Message)
	}
	return &compiledQuery{query: q, language: l, names: q.CaptureNames()}, nil
}

// qualify is the unit's own symbol name, prefixed by any @qualifier the query
// attached. Go's method receiver is the motivating case: a method has no
// enclosing scope node to inherit a name from, so the receiver type is named
// explicitly instead of teaching the core chunker about receivers.
func (u unit) qualify() string {
	if u.qualifier == "" {
		return u.name
	}
	return u.qualifier + "." + u.name
}
