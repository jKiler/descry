package embed

import (
	"reflect"
	"testing"
)

// WordPiece tokenizer behavior. Run: go test ./internal/embed -run WordPiece
//
// Expected ids are derived from the real all-MiniLM-L6-v2 vocab.txt (BERT
// uncased, 30522 entries) vendored in testdata/ — id = zero-based line number.
// They match what the HF tokenizer produces, so the ONNX model sees the exact
// input distribution it was trained on.

func loadTestWordPiece(t *testing.T, maxLen int) *WordPiece {
	t.Helper()
	wp, err := LoadWordPiece("testdata/vocab.txt", maxLen)
	if err != nil {
		t.Fatalf("LoadWordPiece: %v", err)
	}
	return wp
}

func TestWordPiece_KnownSentence(t *testing.T) {
	wp := loadTestWordPiece(t, 256)
	// The exact id sequence the reference BERT tokenizer produces:
	// [CLS] this is an example sentence [SEP]
	enc := wp.Encode("This is an example sentence")

	wantIDs := []int64{101, 2023, 2003, 2019, 2742, 6251, 102}
	if !reflect.DeepEqual(enc.IDs, wantIDs) {
		t.Errorf("IDs = %v, want %v", enc.IDs, wantIDs)
	}
	if want := []int64{1, 1, 1, 1, 1, 1, 1}; !reflect.DeepEqual(enc.Mask, want) {
		t.Errorf("Mask = %v, want %v", enc.Mask, want)
	}
	if want := []int64{0, 0, 0, 0, 0, 0, 0}; !reflect.DeepEqual(enc.TypeIDs, want) {
		t.Errorf("TypeIDs = %v, want %v", enc.TypeIDs, want)
	}
}

func TestWordPiece_SubwordSplitting(t *testing.T) {
	wp := loadTestWordPiece(t, 256)

	// "embeddings" is not a vocab word; greedy longest-match splits it into
	// em ##bed ##ding ##s.
	if got, want := wp.Encode("embeddings").IDs, []int64{101, 7861, 8270, 4667, 2015, 102}; !reflect.DeepEqual(got, want) {
		t.Errorf("embeddings IDs = %v, want %v", got, want)
	}
	// Identifiers subword the same way: descry -> des ##cr ##y.
	if got, want := wp.Encode("descry").IDs, []int64{101, 4078, 26775, 2100, 102}; !reflect.DeepEqual(got, want) {
		t.Errorf("descry IDs = %v, want %v", got, want)
	}
}

func TestWordPiece_PunctuationSplitsBeforeSubwording(t *testing.T) {
	wp := loadTestWordPiece(t, 256)
	// BERT's basic tokenizer isolates punctuation BEFORE wordpiece runs:
	// don't -> don ' t (not don ##' ##t — those exist in the vocab too, so
	// getting this wrong still encodes, just to ids the model never saw).
	if got, want := wp.Encode("don't").IDs, []int64{101, 2123, 1005, 1056, 102}; !reflect.DeepEqual(got, want) {
		t.Errorf("don't IDs = %v, want %v", got, want)
	}
}

func TestWordPiece_LowercasesAndStripsAccents(t *testing.T) {
	wp := loadTestWordPiece(t, 256)
	// uncased BERT: lowercase + strip diacritics. café -> cafe (in vocab);
	// without stripping it would subword or UNK.
	if got, want := wp.Encode("Café").IDs, []int64{101, 7668, 102}; !reflect.DeepEqual(got, want) {
		t.Errorf("Café IDs = %v, want %v", got, want)
	}
}

func TestWordPiece_UnknownBecomesUNK(t *testing.T) {
	wp := loadTestWordPiece(t, 256)
	// ☃ is not in the vocab and no prefix of it is -> single [UNK] (id 100).
	if got, want := wp.Encode("☃").IDs, []int64{101, 100, 102}; !reflect.DeepEqual(got, want) {
		t.Errorf("snowman IDs = %v, want %v", got, want)
	}
}

func TestWordPiece_TruncatesToMaxLen(t *testing.T) {
	wp := loadTestWordPiece(t, 8)
	enc := wp.Encode("this is an example sentence with far too many words to fit")
	if len(enc.IDs) != 8 || len(enc.Mask) != 8 || len(enc.TypeIDs) != 8 {
		t.Fatalf("lengths = %d/%d/%d, want 8 each", len(enc.IDs), len(enc.Mask), len(enc.TypeIDs))
	}
	if enc.IDs[0] != 101 || enc.IDs[7] != 102 {
		t.Errorf("truncated sequence must keep [CLS] first and [SEP] last, got %v", enc.IDs)
	}
}

func TestWordPiece_EmptyInput(t *testing.T) {
	wp := loadTestWordPiece(t, 256)
	// Degenerate but must not panic: just [CLS] [SEP].
	if got, want := wp.Encode("").IDs, []int64{101, 102}; !reflect.DeepEqual(got, want) {
		t.Errorf("empty IDs = %v, want %v", got, want)
	}
}
