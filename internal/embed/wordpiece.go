// WordPiece tokenization for the semantic embedder.
//
// The ONNX model doesn't take text — it takes the integer token ids the
// original BERT training saw. This file reproduces the HF "BertTokenizer"
// pipeline for uncased models: basic-tokenize (lowercase, strip accents,
// isolate punctuation), then greedy longest-match subwording against
// vocab.txt, then wrap in [CLS] ... [SEP].
package embed

import (
	"bufio"
	"fmt"
	"os"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// Encoding is one text in model-ready form. All three slices have equal
// length; the ONNX graph wants each as an int64 tensor of shape [1, len].
type Encoding struct {
	IDs     []int64 // wordpiece token ids, starting [CLS] and ending [SEP]
	Mask    []int64 // attention mask: 1 for every real token (padding is batch-level, added later)
	TypeIDs []int64 // segment ids: all 0 for single-text input
}

// WordPiece tokenizes text against a fixed BERT vocabulary.
type WordPiece struct {
	vocab               map[string]int64
	maxLen              int
	clsID, sepID, unkID int64
}

// maxWordChars matches HF: a single word longer than this is [UNK] outright
// rather than subworded (protects the greedy matcher from degenerate input).
const maxWordChars = 100

// LoadWordPiece reads a vocab.txt (one token per line; id = line number) and
// returns a tokenizer that truncates encodings to maxLen tokens total,
// including [CLS] and [SEP].
func LoadWordPiece(path string, maxLen int) (*WordPiece, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	vocab := make(map[string]int64, 30522)
	scanner := bufio.NewScanner(f)
	for id := int64(0); scanner.Scan(); id++ {
		vocab[scanner.Text()] = id
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	wp := &WordPiece{vocab: vocab, maxLen: maxLen}
	for _, sp := range []struct {
		token string
		id    *int64
	}{{"[CLS]", &wp.clsID}, {"[SEP]", &wp.sepID}, {"[UNK]", &wp.unkID}} {
		id, ok := vocab[sp.token]
		if !ok {
			return nil, fmt.Errorf("vocab %s: missing special token %s", path, sp.token)
		}
		*sp.id = id
	}
	return wp, nil
}

// Encode turns text into the model's input tensors: [CLS] text [SEP],
// truncated to maxLen.
func (w *WordPiece) Encode(text string) Encoding {
	body := w.pieces(text)
	if n := w.maxLen - 2; len(body) > n { // room for [CLS] and [SEP]
		body = body[:n]
	}
	ids := make([]int64, 0, len(body)+2)
	ids = append(ids, w.clsID)
	ids = append(ids, body...)
	ids = append(ids, w.sepID)
	return Encoding{IDs: ids, Mask: ones(len(ids)), TypeIDs: make([]int64, len(ids))}
}

// pieces expands text into wordpiece ids, unbounded — callers apply their own
// length policy, which differs between a single text and a pair.
func (w *WordPiece) pieces(text string) []int64 {
	var ids []int64
	for _, word := range basicTokenize(text) {
		ids = append(ids, w.subword(word)...)
	}
	return ids
}

func ones(n int) []int64 {
	m := make([]int64, n)
	for i := range m {
		m[i] = 1
	}
	return m
}

// subword splits one basic token into wordpiece ids by greedy longest-match:
// take the longest vocab entry at the current position (prefixed "##" past
// the start), repeat from where it ended. Any position with no match — or a
// degenerately long word — collapses the whole word to [UNK].
func (w *WordPiece) subword(word []rune) []int64 {
	if len(word) > maxWordChars {
		return []int64{w.unkID}
	}
	var ids []int64
	for start := 0; start < len(word); {
		end := len(word)
		match := int64(-1)
		for ; end > start; end-- {
			sub := string(word[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if id, ok := w.vocab[sub]; ok {
				match = id
				break
			}
		}
		if match < 0 {
			return []int64{w.unkID}
		}
		ids = append(ids, match)
		start = end
	}
	return ids
}

// basicTokenize mirrors HF's BasicTokenizer for uncased models: lowercase,
// strip accents (NFD, drop combining marks), drop control characters, split
// on whitespace, and isolate every punctuation rune as its own token.
func basicTokenize(text string) [][]rune {
	var words [][]rune
	var cur []rune
	flush := func() {
		if len(cur) > 0 {
			words = append(words, cur)
			cur = nil
		}
	}
	for _, r := range norm.NFD.String(text) {
		switch {
		case unicode.Is(unicode.Mn, r): // combining mark: the accent being stripped
		case r == 0 || r == 0xFFFD || unicode.IsControl(r):
		case unicode.IsSpace(r):
			flush()
		case isPunct(r):
			flush()
			words = append(words, []rune{r})
		default:
			cur = append(cur, unicode.ToLower(r))
		}
	}
	flush()
	return words
}

// isPunct matches HF's _is_punctuation: all non-alphanumeric ASCII (so $, +,
// ^, ` count, which unicode.IsPunct alone would miss) plus Unicode P*.
func isPunct(r rune) bool {
	if r < 0x80 {
		return (r >= '!' && r <= '/') || (r >= ':' && r <= '@') || (r >= '[' && r <= '`') || (r >= '{' && r <= '~')
	}
	return unicode.IsPunct(r)
}
