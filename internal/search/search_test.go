package search

import (
	"math"
	"testing"

	"github.com/jKiler/descry/internal/core"
)

// BM25: BM25 ranks the doc containing the exact term first.
func TestBM25_ExactTermRanksFirst(t *testing.T) {
	docs := []core.Chunk{
		{ID: "a", Content: "the quick brown fox jumps"},
		{ID: "b", Content: "authentication middleware validates the jwt token"},
		{ID: "c", Content: "a recipe for banana bread and walnuts"},
	}
	b := NewBM25()
	b.Index(docs)
	got := b.Search("jwt token", 3)
	if len(got) == 0 {
		t.Fatalf("no results")
	}
	if got[0].Chunk.ID != "b" {
		t.Errorf("top result = %q, want b", got[0].Chunk.ID)
	}
}

// idsOf extracts result IDs, for readable failure messages.
func idsOf(rs []core.SearchResult) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.Chunk.ID
	}
	return out
}

// BM25 — IDF: matching a RARE term must outweigh matching a COMMON one.
// "the" is in every doc (near-zero idf); "pangolin" is in one (high idf).
func TestBM25_IDFRareTermBeatsCommon(t *testing.T) {
	docs := []core.Chunk{
		{ID: "common", Content: "the the the the the the the the"},
		{ID: "rare", Content: "the pangolin"},
		{ID: "d3", Content: "the quick brown fox"},
		{ID: "d4", Content: "the lazy dog sleeps"},
	}
	b := NewBM25()
	b.Index(docs)
	got := b.Search("the pangolin", 4)
	if len(got) == 0 || got[0].Chunk.ID != "rare" {
		t.Fatalf("rare-term doc should rank first (idf downweights 'the'); got %v", idsOf(got))
	}
}

// BM25 — length normalization: same term, same count, the SHORTER doc scores higher.
func TestBM25_ShorterDocScoresHigher(t *testing.T) {
	docs := []core.Chunk{
		{ID: "short", Content: "cache"},
		{ID: "long", Content: "cache and a lot of other unrelated filler words padding this out"},
	}
	b := NewBM25()
	b.Index(docs)
	got := b.Search("cache", 2)
	if len(got) < 2 || got[0].Chunk.ID != "short" {
		t.Fatalf("shorter doc should rank first for the same term; got %v", idsOf(got))
	}
}

// BM25 — term frequency: same length, the doc with MORE occurrences scores higher.
func TestBM25_HigherTermFrequencyScoresHigher(t *testing.T) {
	docs := []core.Chunk{
		{ID: "twice", Content: "cache cache"},
		{ID: "once", Content: "cache miss"},
	}
	b := NewBM25()
	b.Index(docs)
	got := b.Search("cache", 2)
	if len(got) < 2 || got[0].Chunk.ID != "twice" {
		t.Fatalf("more occurrences should rank higher; got %v", idsOf(got))
	}
}

// BM25 — a query whose terms appear in no document returns no results.
func TestBM25_NoMatchReturnsEmpty(t *testing.T) {
	docs := []core.Chunk{
		{ID: "a", Content: "alpha beta gamma"},
		{ID: "b", Content: "delta epsilon"},
	}
	b := NewBM25()
	b.Index(docs)
	if got := b.Search("nonexistent zzz", 5); len(got) != 0 {
		t.Errorf("no matching terms should yield 0 results; got %v", idsOf(got))
	}
}

// RRF: RRF fuses two ranked lists into the expected order.
func TestReciprocalRankFusion_Order(t *testing.T) {
	vec := []string{"x", "y", "z"} // x best here
	lex := []string{"z", "x", "w"} // z best here, x second
	got := ReciprocalRankFusion([][]string{vec, lex}, 60, nil)

	if len(got) < 2 {
		t.Fatalf("want >=2 fused results, got %v", got)
	}
	// x is rank1 + rank2; z is rank3 + rank1. x should edge out z.
	if got[0].ID != "x" {
		t.Errorf("fused[0] = %q, want x", got[0].ID)
	}
	if got[1].ID != "z" {
		t.Errorf("fused[1] = %q, want z", got[1].ID)
	}
}

func TestReciprocalRankFusion_SingleList(t *testing.T) {
	got := ReciprocalRankFusion([][]string{{"a", "b", "c"}}, 60, nil)
	want := []string{"a", "b", "c"}
	for i := range want {
		if i >= len(got) || got[i].ID != want[i] {
			t.Fatalf("got %v, want %v", idsOfFused(got), want)
		}
	}
}

// Score-carrying RRF: the fused result exposes the real Σ 1/(k+rank) score, not
// a rank-derived placeholder. Scores must equal the RRF formula and decrease
// monotonically down the ranking.
func TestReciprocalRankFusion_CarriesScores(t *testing.T) {
	vec := []string{"x", "y", "z"}
	lex := []string{"z", "x", "w"}
	got := ReciprocalRankFusion([][]string{vec, lex}, 60, nil)

	// x appears at vec rank1 and lex rank2: 1/61 + 1/62.
	wantX := 1.0/61 + 1.0/62
	if math.Abs(got[0].Score-wantX) > 1e-9 {
		t.Errorf("x score = %v, want %v", got[0].Score, wantX)
	}
	// y appears only at vec rank2: 1/62.
	var yScore float64
	for _, r := range got {
		if r.ID == "y" {
			yScore = r.Score
		}
	}
	if want := 1.0 / 62; math.Abs(yScore-want) > 1e-9 {
		t.Errorf("y score = %v, want %v", yScore, want)
	}
	// Monotonically non-increasing.
	for i := 1; i < len(got); i++ {
		if got[i].Score > got[i-1].Score {
			t.Errorf("scores not descending at %d: %v > %v", i, got[i].Score, got[i-1].Score)
		}
	}
}

// Weighted fusion scales each list's contribution.
func TestReciprocalRankFusion_Weights(t *testing.T) {
	// Same id "a" at rank1 in both lists; weights double the second list.
	got := ReciprocalRankFusion([][]string{{"a"}, {"a"}}, 60, []float64{1.0, 2.0})
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("got %v, want single id a", idsOfFused(got))
	}
	if want := 1.0/61*1.0 + 1.0/61*2.0; math.Abs(got[0].Score-want) > 1e-9 {
		t.Errorf("weighted score = %v, want %v", got[0].Score, want)
	}
}

// NewHybrid's defaults are the kubernetes-120 sweep optimum (q8: R@10 95.8%,
// MRR 0.720 — see DESIGN.md): RRF k=25, vectors slightly above full weight, the
// chunk-level BM25 at 0.5 beside the whole-file BM25 at 1.0, and max-leaning
// fusion (alpha 0.40) so a strong single-list match isn't buried by consensus.
func TestNewHybrid_Defaults(t *testing.T) {
	h := NewHybrid(nil, nil, nil)
	if h.RRFK != 25 {
		t.Errorf("RRFK = %v, want 25", h.RRFK)
	}
	if h.VecWeight != 1.1 {
		t.Errorf("VecWeight = %v, want 1.1", h.VecWeight)
	}
	if h.LexWeight != 0.5 {
		t.Errorf("LexWeight = %v, want 0.5", h.LexWeight)
	}
	if h.LexFileWeight != 1.0 {
		t.Errorf("LexFileWeight = %v, want 1.0", h.LexFileWeight)
	}
	if h.FuseAlpha != 0.40 {
		t.Errorf("FuseAlpha = %v, want 0.40", h.FuseAlpha)
	}
	if h.CandMult != 8 {
		t.Errorf("CandMult = %v, want 8", h.CandMult)
	}
}

func idsOfFused(rs []FusedResult) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	return out
}

// referenceBM25 scores documents the naive way — scan every doc, recount term
// frequencies — so the postings-based BM25.Search can be checked against an
// independent implementation of the exact same formula on a larger corpus.
func referenceBM25(b *BM25, docs []core.Chunk, query string) map[string]float64 {
	tok := func(s string) []string { return tokenizeForRef(s) }
	docToks := make([][]string, len(docs))
	total := 0
	for i, d := range docs {
		t := tok(d.Content) // PathTokens off in this test (Path/Symbol empty)
		docToks[i] = t
		total += len(t)
	}
	avg := float64(total) / float64(len(docs))
	df := map[string]int{}
	for _, ts := range docToks {
		seen := map[string]bool{}
		for _, w := range ts {
			if !seen[w] {
				seen[w] = true
				df[w]++
			}
		}
	}
	idf := func(t string) float64 {
		d := float64(df[t])
		return math.Log(1 + (float64(len(docs))-d+0.5)/(d+0.5))
	}
	scores := map[string]float64{}
	for _, qt := range tok(query) {
		for i, ts := range docToks {
			var f float64
			for _, w := range ts {
				if w == qt {
					f++
				}
			}
			if f == 0 {
				continue
			}
			dl := float64(len(ts))
			denom := f + b.K1*(1-b.B+b.B*dl/avg)
			scores[docs[i].ID] += idf(qt) * (f * (b.K1 + 1)) / denom
		}
	}
	return scores
}

func tokenizeForRef(s string) []string {
	// The reference uses the same tokenizer the index does.
	return tokenizeExported(s)
}

// Postings-based scoring must equal the brute-force reference on a non-trivial
// corpus, not just the small fixtures.
func TestBM25_MatchesBruteForceReference(t *testing.T) {
	vocab := []string{"cache", "store", "vector", "index", "query", "token", "graph", "search", "embed", "hybrid", "the", "a", "of"}
	rng := newSeededRand(42)
	docs := make([]core.Chunk, 300)
	for i := range docs {
		n := 5 + rng(40)
		var sb []string
		for j := 0; j < n; j++ {
			sb = append(sb, vocab[rng(len(vocab))])
		}
		docs[i] = core.Chunk{ID: idFor(i), Content: joinWords(sb)}
	}
	b := &BM25{K1: 1.5, B: 0.75} // PathTokens off: content only
	b.Index(docs)

	for _, q := range []string{"cache vector", "the query token", "graph search embed hybrid", "of a the"} {
		got := b.Search(q, len(docs))
		want := referenceBM25(b, docs, q)
		if len(got) != countPositive(want) {
			t.Fatalf("q=%q: result count %d, want %d", q, len(got), countPositive(want))
		}
		for _, r := range got {
			if math.Abs(r.Score-want[r.Chunk.ID]) > 1e-9 {
				t.Errorf("q=%q doc=%s: score %v, want %v", q, r.Chunk.ID, r.Score, want[r.Chunk.ID])
			}
		}
	}
}

func idFor(i int) string { return "d" + itoa(i) }
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}
func joinWords(w []string) string {
	out := ""
	for i, s := range w {
		if i > 0 {
			out += " "
		}
		out += s
	}
	return out
}
func newSeededRand(seed uint64) func(n int) int {
	s := seed
	return func(n int) int {
		s ^= s << 13
		s ^= s >> 7
		s ^= s << 17
		return int(s % uint64(n))
	}
}
func countPositive(m map[string]float64) int {
	c := 0
	for _, v := range m {
		if v > 0 {
			c++
		}
	}
	return c
}

func tokenizeExported(s string) []string { return tokenizeShim(s) }
