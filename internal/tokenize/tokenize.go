// Package tokenize splits source text and identifiers into search tokens.
//
// Splitting camelCase and snake_case means a search for "depends" matches the
// symbol "getDependsOn" — a small trick with an outsized effect on recall. Used
// by the BM25 lexical ranker.
package tokenize

import (
	"strings"
	"unicode"
)

// Tokenize lowercases text and splits it into word tokens, breaking on:
//   - non-alphanumeric characters (spaces, punctuation, "_"),
//   - camelCase boundaries (lower->Upper, e.g. "getDependsOn"),
//   - digit boundaries are optional (keep it simple).
//
// Behavior (see tokenize_test.go):
//
//	Tokenize("getDependsOn")  -> ["get", "depends", "on"]
//	Tokenize("user_id_map")   -> ["user", "id", "map"]
//	Tokenize("HTTPServer")    -> ["http", "server"]  (acronym run)
//	All tokens lowercased; no empty tokens.
func Tokenize(text string) []string {
	// Step 1: break on any non-letter/digit run (spaces, punctuation, "_").
	// Do NOT lowercase yet — camelCase boundaries are still needed in step 2.
	fields := strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})

	// Step 2: split each field on camelCase / acronym boundaries, then lowercase.
	var tokens []string
	for _, field := range fields {
		tokens = append(tokens, splitCamel(field)...)
	}
	return tokens
}

// splitCamel breaks one field on case boundaries and lowercases the pieces:
//
//	"getDependsOn" -> get, depends, on
//	"HTTPServer"   -> http, server
//
// Two boundaries: a lowercase/digit followed by an uppercase ("getD|epends"),
// and an uppercase followed by an uppercase-then-lowercase, which ends an
// acronym run ("HTTP|Server").
func splitCamel(field string) []string {
	runes := []rune(field)
	var out []string
	start := 0
	for i := 1; i < len(runes); i++ {
		prev, curr := runes[i-1], runes[i]

		lowerToUpper := isLowerOrDigit(prev) && unicode.IsUpper(curr)
		acronymEnd := unicode.IsUpper(prev) && unicode.IsUpper(curr) &&
			i+1 < len(runes) && unicode.IsLower(runes[i+1])

		if lowerToUpper || acronymEnd {
			out = append(out, strings.ToLower(string(runes[start:i])))
			start = i
		}
	}
	return append(out, strings.ToLower(string(runes[start:])))
}

func isLowerOrDigit(r rune) bool {
	return unicode.IsLower(r) || unicode.IsDigit(r)
}
