package chunk

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/jKiler/descry/internal/core"
	"github.com/jKiler/descry/internal/lang"
	ts "github.com/tree-sitter/go-tree-sitter"
)

// assemble turns captured units into the final chunk list. It does three
// things, in order:
//
//  1. Cover. Chunk units are laid down in file order; every byte they leave
//     uncovered becomes a gap chunk attributed to its innermost enclosing
//     scope. This is what makes "never silently vanish from the index" a
//     property of the algorithm rather than a hope about the queries.
//  2. Qualify. Each chunk's symbol is prefixed by the names of the scopes
//     containing it, so a method reads as "QuerySet.filter", not "filter".
//  3. Fit. A chunk whose embedded text exceeds the token budget is split into
//     overlapping windows cut on statement boundaries, each carrying the
//     declaration's header.
func (c *TSChunker) assemble(path string, src []byte, p *lang.Pack, units []unit, root *ts.Node) []core.Chunk {
	// Chunk ids are content-keyed, and duplicates within one file are
	// disambiguated in emission order, so the map is per-file.
	seen := map[string]int{}
	var chunkUnits, scopes, drops []unit
	for _, u := range units {
		switch {
		case u.drop:
			drops = append(drops, u)
		case u.scope:
			scopes = append(scopes, u)
		default:
			chunkUnits = append(chunkUnits, u)
		}
	}
	sortUnits(chunkUnits)
	sortUnits(scopes)
	sortUnits(drops)

	// A dropped region counts as covered: the cover exists so nothing is lost by
	// accident, not so a deliberate exclusion comes back as a gap chunk.
	chunkUnits = append(chunkUnits, drops...)
	sortUnits(chunkUnits)

	lines := newLineMap(src)
	var out []core.Chunk

	emit := func(start, end uint, name string, sigEnd uint, node *ts.Node) {
		if strings.TrimSpace(string(src[start:end])) == "" {
			return
		}
		qualified := qualify(scopes, start, end, name)
		out = append(out, c.fit(path, src, p, lines, start, end, sigEnd, qualified, node, root, seen)...)
	}

	var cursor uint
	for _, u := range chunkUnits {
		if u.start > cursor {
			c.emitGaps(&out, path, src, p, lines, scopes, root, cursor, u.start, seen)
		}
		if !u.drop {
			emit(u.start, u.end, u.qualify(), u.sigEnd, u.node)
		}
		if u.end > cursor {
			cursor = u.end
		}
	}
	if cursor < uint(len(src)) {
		c.emitGaps(&out, path, src, p, lines, scopes, root, cursor, uint(len(src)), seen)
	}

	sort.SliceStable(out, func(i, j int) bool { return out[i].StartLine < out[j].StartLine })
	return out
}

// emitGaps turns an uncovered byte range into chunks. The range is split on
// blank lines — the same unit a reader sees — but only at blank lines that are
// *safe*: a blank line inside a module docstring or a multi-line string literal
// is not a boundary, it is content, and splitting there shreds one paragraph of
// documentation into eight useless chunks. Each piece is attributed to the
// innermost scope containing it, so a class's fields carry the class name even
// though no query captured them.
func (c *TSChunker) emitGaps(out *[]core.Chunk, path string, src []byte, p *lang.Pack, lines *lineMap, scopes []unit, root *ts.Node, from, to uint, seen map[string]int) {
	for _, piece := range c.gapPieces(src, root, scopes, from, to) {
		body := string(src[piece.start:piece.end])
		if !worthIndexing(body, c.MinGapLines) {
			continue
		}
		name := scopeNameAt(scopes, piece.start, piece.end)
		*out = append(*out, c.fit(path, src, p, lines, piece.start, piece.end, piece.start, name, nil, root, seen)...)
	}
}

// gapPieces splits [from,to) at the blank lines that are safe cut points, then
// merges neighbours back together while they share a scope and fit the budget.
//
// The merge is what keeps the cover from producing confetti. A three-line class
// whose docstring and `pass` are separated by a blank line is one thing, not
// two; a file's import block is one thing, not six. Splitting them costs
// retrieval twice over — each fragment is too small to carry meaning, and the
// fragments compete with each other for the same rank.
func (c *TSChunker) gapPieces(src []byte, root *ts.Node, scopes []unit, from, to uint) []span {
	var pieces []span
	start := from
	for _, b := range blankLineStarts(src, from, to) {
		if !safeCut(root, b) {
			continue
		}
		if b > start {
			pieces = append(pieces, span{start, b})
		}
		start = b
	}
	if start < to {
		pieces = append(pieces, span{start, to})
	}

	var out []span
	for _, p := range pieces {
		if n := len(out); n > 0 &&
			scopeNameAt(scopes, out[n-1].start, out[n-1].end) == scopeNameAt(scopes, p.start, p.end) &&
			estimateTokens(string(src[out[n-1].start:p.end])) <= c.MaxTokens {
			out[n-1].end = p.end
			continue
		}
		out = append(out, p)
	}
	return out
}

// blankLineStarts returns the offsets in [from,to) at which a run of blank
// lines ends — that is, where the next non-blank paragraph begins.
func blankLineStarts(src []byte, from, to uint) []uint {
	var out []uint
	lineStart := from
	prevBlank := false
	for i := from; i <= to; i++ {
		if i == to || src[i] == '\n' {
			blank := strings.TrimSpace(string(src[lineStart:i])) == ""
			if prevBlank && !blank {
				out = append(out, lineStart)
			}
			prevBlank = blank
			lineStart = i + 1
		}
	}
	return out
}

// safeCut reports whether a byte offset is a place a chunk may begin or end
// without tearing a syntactic unit.
//
// The rule is recursive and needs no knowledge of any language: descend to the
// smallest node strictly containing the offset. If the offset falls between two
// of that node's named children — or in the node's own keyword and punctuation
// — the cut is between units and is safe. If it reaches a node with no named
// children, the cut is inside a leaf (a string literal, an identifier, a
// comment) and is not.
func safeCut(n *ts.Node, off uint) bool {
	if off <= n.StartByte() || off >= n.EndByte() {
		return true // outside this node entirely
	}
	for i := uint(0); i < n.NamedChildCount(); i++ {
		ch := n.NamedChild(i)
		if off > ch.StartByte() && off < ch.EndByte() {
			return safeCut(ch, off)
		}
	}
	return n.NamedChildCount() > 0
}

// worthIndexing rejects an uncovered region that is only punctuation or is
// shorter than the minimum and carries no identifier: a closing brace on its
// own line is not something anyone searches for.
func worthIndexing(body string, minLines int) bool {
	trimmed := strings.TrimSpace(body)
	if trimmed == "" {
		return false
	}
	// Unicode-aware on purpose: an ASCII-only test would judge a comment or an
	// identifier written in any non-Latin script to be wordless and drop the
	// region from the cover, which is precisely the silent disappearance the
	// cover exists to prevent.
	hasWord := strings.ContainsFunc(trimmed, func(r rune) bool {
		return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
	})
	if !hasWord {
		return false
	}
	return strings.Count(trimmed, "\n")+1 >= minLines
}

// qualify prefixes a symbol with the names of the scopes containing it.
func qualify(scopes []unit, start, end uint, name string) string {
	var parts []string
	for _, s := range scopes {
		if s.name != "" && s.start <= start && s.end >= end {
			parts = append(parts, s.name)
		}
	}
	if name != "" {
		parts = append(parts, name)
	}
	return strings.Join(parts, ".")
}

// scopeNameAt names the innermost scope containing a range, for gap chunks.
func scopeNameAt(scopes []unit, start, end uint) string {
	best := ""
	var bestSize uint = ^uint(0)
	for _, s := range scopes {
		if s.name == "" || s.start > start || s.end < end {
			continue
		}
		if size := s.end - s.start; size < bestSize {
			best, bestSize = s.name, size
		}
	}
	return best
}

// fit turns one span into one chunk, or into several overlapping windows when
// its embedded text would not fit the model's input.
//
// Windows keep the two representations distinct, which is the point: the raw
// Content stays a clean, contiguous slice of the file (what an agent reads),
// while the repeated header goes only into EmbedText (what the model sees). A
// window is therefore still a well-formed piece of source, and cutting on
// statement boundaries keeps it structurally sound.
func (c *TSChunker) fit(path string, src []byte, p *lang.Pack, lines *lineMap, start, end, sigEnd uint, symbol string, node, root *ts.Node, seen map[string]int) []core.Chunk {
	body := string(src[start:end])
	headerBudget := c.MaxTokens
	if !c.CapHeader {
		headerBudget = 0 // 0 disables the cap; see enrichHeader
	}
	header := enrichHeader(p.Name, path, symbol, signatureText(src, start, sigEnd), headerBudget)

	if estimateTokens(header)+estimateTokens(body) <= c.MaxTokens {
		return []core.Chunk{c.chunk(path, lines, start, end, symbol, body, header+body, seen)}
	}

	cuts := statementCuts(node, start, end)
	budget := c.MaxTokens - estimateTokens(header)
	if budget < 32 { // a header this large leaves no room; keep the chunk whole
		return []core.Chunk{c.chunk(path, lines, start, end, symbol, body, header+body, seen)}
	}
	windows := window(src, root, start, end, cuts, budget, c.OverlapStatements)
	out := make([]core.Chunk, 0, len(windows))
	for i, w := range windows {
		wb := string(src[w.start:w.end])
		sym := symbol
		if len(windows) > 1 && symbol != "" {
			sym = fmt.Sprintf("%s~%d", symbol, i+1)
		}
		out = append(out, c.chunk(path, lines, w.start, w.end, sym, wb, header+wb, seen))
	}
	return out
}

func (c *TSChunker) chunk(path string, lines *lineMap, start, end uint, symbol, body, embed string, seen map[string]int) core.Chunk {
	sl, el := lines.line(start), lines.line(lastByte(end))
	content := strings.Trim(body, "\n")
	return core.Chunk{
		ID:        IDFor(path, content, seen),
		Path:      path,
		StartLine: sl,
		EndLine:   el,
		StartByte: int(start),
		EndByte:   int(end),
		Symbol:    symbol,
		Content:   content,
		EmbedText: embed,
	}
}

func lastByte(end uint) uint {
	if end == 0 {
		return 0
	}
	return end - 1
}

// enrichHeader is the enriched half of a chunk's two representations: the
// language, the module path, and the qualified symbol with its signature, in
// words a sentence embedder understands after WordPiece. The raw source that
// follows is what an agent reads back.
//
// The header competes with the code for the same token budget, and it must not
// win. Java is what forced this to be explicit: its package directories are
// deep enough that the path alone cost 70 of 220 tokens, so a 35-line method
// was cut into seven windows of five lines each — each one carrying more path
// than code. The header is therefore trimmed to a fraction of the budget,
// shedding leading path segments first (the file's own name and its nearest
// package are the informative part) and then the signature.
func enrichHeader(langName, path, symbol, signature string, budget int) string {
	if budget <= 0 {
		return buildHeader(langName, path, symbol, signature) // cap disabled
	}
	max := budget / headerShare
	for {
		h := buildHeader(langName, path, symbol, signature)
		if estimateTokens(h) <= max {
			return h
		}
		if shorter, ok := dropLeadingSegment(path); ok {
			path = shorter
			continue
		}
		if signature != "" {
			signature = ""
			continue
		}
		return h // basename and symbol alone; nothing left to shed
	}
}

// headerShare is the divisor bounding the header's slice of the token budget.
// A quarter leaves three quarters for code, which is the point of the chunk.
const headerShare = 4

func buildHeader(langName, path, symbol, signature string) string {
	var b strings.Builder
	b.WriteString(langName)
	b.WriteByte(' ')
	b.WriteString(path)
	b.WriteByte('\n')
	if symbol != "" {
		b.WriteString(symbol)
		b.WriteByte('\n')
	}
	if signature != "" {
		b.WriteString(signature)
		b.WriteByte('\n')
	}
	return b.String()
}

// dropLeadingSegment removes the outermost directory of a path, keeping the
// basename. It reports false once only the basename is left.
func dropLeadingSegment(path string) (string, bool) {
	i := strings.IndexByte(path, '/')
	if i < 0 || i+1 >= len(path) {
		return path, false
	}
	return path[i+1:], true
}

// sigMaxBytes bounds the collapsed signature. It is a byte budget rather than a
// rune count because what it protects is the token budget, and the tokenizer
// charges bytes.
const sigMaxBytes = 200

// signatureText is the declaration header — everything from the start of the
// declaration through the end of its signature — collapsed to one line.
func signatureText(src []byte, start, sigEnd uint) string {
	if sigEnd <= start || sigEnd > uint(len(src)) {
		return ""
	}
	s := strings.Join(strings.Fields(string(src[start:sigEnd])), " ")
	// Truncate on a rune boundary. Slicing bytes would cut a multi-byte
	// character in half, and this string is concatenated into the text handed to
	// the tokenizer — a signature ending in half a character is malformed UTF-8
	// in the embedded representation of every chunk of that declaration.
	if len(s) > sigMaxBytes {
		s = s[:sigMaxBytes]
		for len(s) > 0 && !utf8.ValidString(s) {
			s = s[:len(s)-1]
		}
	}
	return s
}

// statementCuts returns byte offsets inside [start,end) that are safe places to
// end a window: the boundaries between the statements of the node's body. It is
// language-agnostic — the body is simply the node's largest named child, and
// its named children are the statements.
//
// The *first* statement's start is deliberately not a cut point. Cutting there
// would emit a window holding nothing but the declaration's signature, which is
// both useless to retrieve and the "signature separated from body" failure the
// structural check exists to catch.
func statementCuts(node *ts.Node, start, end uint) []uint {
	if node == nil {
		return nil
	}
	body := largestNamedChild(node)
	if body == nil {
		return nil
	}
	var cuts []uint
	for i := uint(1); i < body.NamedChildCount(); i++ {
		ch := body.NamedChild(i)
		if b := ch.StartByte(); b > start && b < end {
			cuts = append(cuts, b)
		}
	}
	return cuts
}

func largestNamedChild(n *ts.Node) *ts.Node {
	var best *ts.Node
	var bestSize uint
	for i := uint(0); i < n.NamedChildCount(); i++ {
		ch := n.NamedChild(i)
		if size := ch.EndByte() - ch.StartByte(); size > bestSize {
			best, bestSize = ch, size
		}
	}
	return best
}

// window packs [start,end) into consecutive ranges under the token budget,
// cutting only at the given boundaries and repeating overlap boundaries between
// neighbours. With no usable cut points it falls back to line boundaries, which
// still never split a line.
func window(src []byte, root *ts.Node, start, end uint, cuts []uint, budget, overlap int) []span {
	bounds := append([]uint{start}, cuts...)
	bounds = append(bounds, end)
	if len(bounds) <= 2 {
		bounds = lineBounds(src, root, start, end)
	}

	var out []span
	i := 0
	for i < len(bounds)-1 {
		j := i + 1
		for j < len(bounds)-1 && estimateTokens(string(src[bounds[i]:bounds[j+1]])) <= budget {
			j++
		}
		out = append(out, span{bounds[i], bounds[j]})
		if j >= len(bounds)-1 {
			break
		}
		next := j - overlap
		if next <= i {
			next = i + 1 // always make progress, even if one statement busts the budget
		}
		i = next
	}
	if len(out) == 0 {
		out = append(out, span{start, end})
	}
	return out
}

// span is a byte range in the source.
type span struct{ start, end uint }

// lineBounds returns the line starts in [start,end) that are safe cut points —
// the finest cut a window may use when a node offers no statement structure.
// When nothing is safe (a single enormous string literal), every line start is
// offered anyway: cutting prose mid-paragraph is worse than nothing, but losing
// the tail to truncation is worse still.
func lineBounds(src []byte, root *ts.Node, start, end uint) []uint {
	var safe, all []uint
	for i := start; i < end; i++ {
		if src[i] != '\n' || i+1 >= end {
			continue
		}
		all = append(all, i+1)
		if root == nil || safeCut(root, i+1) {
			safe = append(safe, i+1)
		}
	}
	if len(safe) == 0 {
		safe = all
	}
	return append(append([]uint{start}, safe...), end)
}

// estimateTokens approximates the WordPiece length of a string without running
// the tokenizer — which lives behind the embedder, and would make chunking
// depend on model provisioning.
//
// It models what WordPiece actually does rather than guessing a bytes-per-token
// ratio: split on whitespace and punctuation, charge each punctuation character
// one token, and charge each word ceil(len/4) subword pieces. Measured against
// the real tokenizer over 4000 chunk-sized samples from the Python, Rust, C and
// TypeScript corpora, this lands at a median ratio of 0.98 with p05 0.82 and
// p95 1.19 — against 1.05 median but 0.67/1.46 spread for a chars/3 rule, which
// both truncated long chunks and windowed short ones. MaxTokens is set below
// the model's real ceiling to absorb the remaining spread.
func estimateTokens(s string) int {
	n, wordLen := 0, 0
	flush := func() {
		if wordLen > 0 {
			n += (wordLen + 3) / 4
			wordLen = 0
		}
	}
	for _, r := range s {
		switch {
		case r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9'):
			wordLen++
		case r == ' ' || r == '\t' || r == '\n' || r == '\r':
			flush()
		default:
			flush()
			n++
		}
	}
	flush()
	return n
}

// lineMap converts byte offsets to 1-based line numbers in one pass.
type lineMap struct{ starts []uint }

func newLineMap(src []byte) *lineMap {
	starts := []uint{0}
	for i, b := range src {
		if b == '\n' {
			starts = append(starts, uint(i)+1)
		}
	}
	return &lineMap{starts: starts}
}

func (m *lineMap) line(off uint) int {
	i := sort.Search(len(m.starts), func(i int) bool { return m.starts[i] > off })
	if i == 0 {
		return 1
	}
	return i
}
