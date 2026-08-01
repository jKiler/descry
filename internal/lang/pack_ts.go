package lang

import (
	ts "github.com/tree-sitter/go-tree-sitter"
	jsgr "github.com/tree-sitter/tree-sitter-javascript/bindings/go"
	tsgr "github.com/tree-sitter/tree-sitter-typescript/bindings/go"
)

// TypeScript and JavaScript are separate packs because they are separate
// tree-sitter grammars, even though they share every file-classification
// convention. TSX needs the TSX dialect, which is why .tsx is claimed
// separately from .ts.
//
// The arrow-function patterns matter more here than anywhere else: in modern
// TS/JS a large share of the real functions are `const f = (…) => {…}`, and a
// chunker that only knows `function` declarations loses them into module-level
// gap chunks.
var (
	TypeScript = &Pack{
		Name:             "typescript",
		Exts:             []string{".ts", ".mts", ".cts"},
		TestBases:        tsTestBases,
		TestDirs:         tsTestDirs,
		GeneratedMarkers: tsGenerated,
		WrapKinds:        []string{"export_statement"},
		Grammar:          func() *ts.Language { return ts.NewLanguage(tsgr.LanguageTypescript()) },
		Query:            tsQuery,
	}
	TSX = &Pack{
		Name:             "tsx",
		Exts:             []string{".tsx"},
		TestBases:        tsTestBases,
		TestDirs:         tsTestDirs,
		GeneratedMarkers: tsGenerated,
		WrapKinds:        []string{"export_statement"},
		Grammar:          func() *ts.Language { return ts.NewLanguage(tsgr.LanguageTSX()) },
		Query:            tsQuery,
	}
	JavaScript = &Pack{
		Name:             "javascript",
		Exts:             []string{".js", ".jsx", ".mjs", ".cjs"},
		TestBases:        tsTestBases,
		TestDirs:         tsTestDirs,
		GeneratedMarkers: tsGenerated,
		WrapKinds:        []string{"export_statement"},
		Grammar:          func() *ts.Language { return ts.NewLanguage(jsgr.Language()) },
		Query:            jsQuery,
	}
)

// commonJSQuery covers the constructs both dialects share. The class
// declaration itself is not here: the two grammars disagree on the type of a
// class's name node (`identifier` in JavaScript, `type_identifier` in
// TypeScript), and a pattern naming the wrong one fails the whole query to
// compile — which silently degrades the entire language to blank-line
// splitting. TestPackQueriesCompile exists because of exactly that failure.
const commonJSQuery = `
(function_declaration name: (identifier) @name body: (statement_block) @body) @chunk
(generator_function_declaration name: (identifier) @name body: (statement_block) @body) @chunk
(class_body (method_definition name: (property_identifier) @name body: (statement_block) @body) @chunk)
(lexical_declaration (variable_declarator
    name: (identifier) @name
    value: [(arrow_function) (function_expression)] @body)) @chunk
(variable_declaration (variable_declarator
    name: (identifier) @name
    value: [(arrow_function) (function_expression)] @body)) @chunk
`

const jsQuery = commonJSQuery + `
(class_declaration name: (identifier) @name body: (class_body) @body) @scope
`

// tsQuery adds the type-level declarations only TypeScript has.
const tsQuery = commonJSQuery + `
(class_declaration name: (type_identifier) @name body: (class_body) @body) @scope
(abstract_class_declaration name: (type_identifier) @name body: (class_body) @body) @scope
(interface_declaration name: (type_identifier) @name body: (interface_body) @body) @chunk
(type_alias_declaration name: (type_identifier) @name) @chunk
(enum_declaration name: (identifier) @name body: (enum_body) @body) @chunk
(class_body (method_signature name: (property_identifier) @name) @chunk)
(function_signature name: (identifier) @name) @chunk
`

var (
	tsTestBases = []string{
		"*.test.ts", "*.test.tsx", "*.test.js", "*.test.jsx", "*.test.mjs",
		"*.spec.ts", "*.spec.tsx", "*.spec.js", "*.spec.jsx", "*.spec.mjs",
	}
	tsTestDirs  = []string{"__tests__", "__mocks__", "__snapshots__", "e2e"}
	tsGenerated = []string{"@generated", "DO NOT EDIT"}
)

func init() {
	Register(TypeScript)
	Register(TSX)
	Register(JavaScript)
}
