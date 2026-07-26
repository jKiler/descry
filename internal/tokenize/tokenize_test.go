package tokenize

import (
	"reflect"
	"testing"
)

// Tokenizer behavior. Run: go test ./internal/tokenize
func TestTokenize(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"getDependsOn", []string{"get", "depends", "on"}},
		{"user_id_map", []string{"user", "id", "map"}},
		{"where is auth handled", []string{"where", "is", "auth", "handled"}},
		{"parseJSON()", []string{"parse", "json"}},
	}
	for _, c := range cases {
		if got := Tokenize(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("Tokenize(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestTokenize_MatchesIdentifierSubword(t *testing.T) {
	// The whole point: a plain-word query token appears inside a split symbol.
	toks := Tokenize("getDependsOn")
	found := false
	for _, tk := range toks {
		if tk == "depends" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected 'depends' among %v", toks)
	}
}
