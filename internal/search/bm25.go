// Package search provides the retrieval layer: lexical (BM25, this file), rank
// fusion (RRF, fusion.go), and the hybrid combiner (hybrid.go).
//
// BM25 is the classic keyword ranking function. It shines where embeddings are
// fuzzy — exact identifiers — so the hybrid retriever fuses it with vector
// search rather than choosing one.
package search

import (
	"math"
	"sort"

	"github.com/jKiler/descry/internal/core"
	"github.com/jKiler/descry/internal/tokenize"
)

// posting is one document's entry in a term's postings list: the document id
// (an index into chunks) and the term's frequency in that document.
type posting struct {
	doc uint32
	tf  uint32
}

// BM25 ranks chunks by keyword relevance over an inverted index (term ->
// postings), so a query scores only the documents that actually contain its
// terms rather than scanning the whole corpus.
//
// Standard parameters: k1 controls term-frequency saturation (~1.2–2.0),
// b controls length normalization (~0.75).
type BM25 struct {
	K1 float64
	B  float64

	// PathTokens folds each chunk's file path (and symbol) into its token bag,
	// so a query naming a component ("bm25 scoring", "sqlite store") gets
	// lexical credit even when the body text never spells the name out. Path
	// segments like "internal" or "go" are near-universal, so their idf ≈ 0 —
	// they add nothing; distinctive segments ("bm25", "wordpiece") score high.
	PathTokens bool

	// Inverted index, populated by Index (or restored by DecodeBM25).
	chunks   []core.Chunk         // doc id (index) -> chunk
	postings map[string][]posting // term -> the docs containing it, with tf
	docLen   []int                // doc id -> token count
	avgLen   float64              // average document length in tokens
}

// NewBM25 returns a BM25 ranker with standard defaults.
func NewBM25() *BM25 { return &BM25{K1: 1.5, B: 0.75, PathTokens: true} }

// Index builds the inverted index from chunks: it tokenizes each chunk's content
// (plus its path and symbol when PathTokens is set), records per-doc lengths, and
// appends a posting to each term's list. Document frequency is len(postings), so
// it isn't stored separately.
func (b *BM25) Index(chunks []core.Chunk) {
	b.chunks = chunks
	b.postings = make(map[string][]posting)
	b.docLen = make([]int, len(chunks))

	totalTokens := 0
	for i, chunk := range chunks {
		tokens := tokenize.Tokenize(chunk.Content)
		if b.PathTokens {
			tokens = append(tokens, tokenize.Tokenize(chunk.Path)...)
			tokens = append(tokens, tokenize.Tokenize(chunk.Symbol)...)
		}
		b.docLen[i] = len(tokens)
		totalTokens += len(tokens)

		freq := make(map[string]uint32, len(tokens))
		for _, tok := range tokens {
			freq[tok]++
		}
		for term, tf := range freq {
			b.postings[term] = append(b.postings[term], posting{doc: uint32(i), tf: tf})
		}
	}
	if len(chunks) > 0 {
		b.avgLen = float64(totalTokens) / float64(len(chunks))
	}
}

// Search returns the topK chunks by BM25 score for the query. Each query term's
// postings are walked once, adding that term's contribution to every document it
// occurs in; documents no term touches are never scored. Results are sorted
// descending, with the chunk ID as a deterministic tie-break.
func (b *BM25) Search(query string, topK int) []core.SearchResult {
	qTerms := tokenize.Tokenize(query)

	scores := make(map[uint32]float64)
	for _, t := range qTerms {
		pl := b.postings[t]
		if len(pl) == 0 {
			continue // term absent from the corpus — contributes nothing
		}
		idf := b.idf(len(pl))
		for _, p := range pl {
			scores[p.doc] += idf * b.tfNorm(p.tf, b.docLen[p.doc])
		}
	}

	results := make([]core.SearchResult, 0, len(scores))
	for doc, s := range scores {
		results = append(results, core.SearchResult{Chunk: b.chunks[doc], Score: s})
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Score != results[j].Score {
			return results[i].Score > results[j].Score
		}
		return results[i].Chunk.ID < results[j].Chunk.ID // deterministic ties
	})

	topK = min(max(topK, 0), len(results))
	out := make([]core.SearchResult, topK)
	copy(out, results)
	return out
}

// tfNorm is the term-frequency component of BM25 for a term occurring tf times in
// a document of the given length:
//
//	f·(k1+1) / ( f + k1·(1 - b + b·|d|/avgLen) )
func (b *BM25) tfNorm(tf uint32, docLen int) float64 {
	f := float64(tf)
	denom := f + b.K1*(1-b.B+b.B*float64(docLen)/b.avgLen)
	return f * (b.K1 + 1) / denom
}

// idf is the inverse document frequency for a term whose postings list has df
// entries: ln(1 + (N - df + 0.5)/(df + 0.5)). The 1+ is OUTSIDE the division,
// which keeps idf non-negative.
func (b *BM25) idf(df int) float64 {
	n := float64(len(b.chunks))
	d := float64(df)
	return math.Log(1 + (n-d+0.5)/(d+0.5))
}

// FileDocs collapses chunks into one synthetic document per file, in first-
// appearance order: the contents concatenate, and every chunk symbol joins a
// space-separated bag so PathTokens still credits them. Indexing these with a
// BM25 gives a lexical ranking over whole files — a cleaner file signal than
// collapsing a chunk ranking, because term frequency and document length are
// then measured on the real unit of retrieval (see Hybrid.SearchFiles).
func FileDocs(chunks []core.Chunk) []core.Chunk {
	order := make(map[string]int)
	var docs []core.Chunk
	for _, c := range chunks {
		i, ok := order[c.Path]
		if !ok {
			i = len(docs)
			order[c.Path] = i
			docs = append(docs, core.Chunk{ID: c.Path, Path: c.Path, StartLine: c.StartLine})
		}
		d := &docs[i]
		if d.Content != "" {
			d.Content += "\n"
		}
		d.Content += c.Content
		if c.Symbol != "" {
			if d.Symbol != "" {
				d.Symbol += " "
			}
			d.Symbol += c.Symbol
		}
		d.EndLine = max(d.EndLine, c.EndLine)
	}
	return docs
}
