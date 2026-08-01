package lang

import (
	ts "github.com/tree-sitter/go-tree-sitter"
	javagr "github.com/tree-sitter/tree-sitter-java/bindings/go"
)

// Java. Maven and Gradle both put test sources under `src/test/...`, which is
// the reliable signal; the `*Test.java` / `*Tests.java` / `*IT.java` name
// conventions catch the rest.
//
// Classes, interfaces and enums are scopes rather than chunks: in Java the
// retrievable unit is almost always a method, and a class that became one chunk
// would be a 600-line blob whose tail the embedder never sees. The class's own
// declaration — its javadoc, its `implements` list, its fields — is covered by
// the gap chunk the cover leaves behind, which is exactly the class summary
// someone searching for the type wants.
var Java = &Pack{
	Name:             "java",
	Exts:             []string{".java"},
	TestBases:        []string{"*Test.java", "*Tests.java", "*IT.java", "*TestCase.java"},
	TestPaths:        []string{"src/test"},
	GeneratedMarkers: []string{"@Generated", "DO NOT EDIT", "This file was automatically generated"},
	Grammar:          func() *ts.Language { return ts.NewLanguage(javagr.Language()) },
	Query: `
(class_declaration name: (identifier) @name body: (class_body) @body) @scope
(interface_declaration name: (identifier) @name body: (interface_body) @body) @scope
(enum_declaration name: (identifier) @name body: (enum_body) @body) @scope
(annotation_type_declaration name: (identifier) @name) @scope

(method_declaration name: (identifier) @name body: (block) @body) @chunk
(method_declaration name: (identifier) @name !body) @chunk
(constructor_declaration name: (identifier) @name body: (constructor_body) @body) @chunk
(record_declaration name: (identifier) @name) @chunk
`,
}

func init() { Register(Java) }
