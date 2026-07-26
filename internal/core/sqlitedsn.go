package core

import (
	"path/filepath"
	"strings"
)

// dsnPathEscaper escapes the bytes a SQLite file: URI reads as syntax rather
// than filename: '?' starts the query, '#' the fragment, '%HH' an escape.
var dsnPathEscaper = strings.NewReplacer("%", "%25", "?", "%3f", "#", "%23")

// SQLiteDSN builds the connection string for a SQLite database at path, with
// optional query parameters ("mode=ro"). Every sqlite open in descry — the
// index, the embed cache, diagnostics — must go through it, because these
// databases live inside the repository being indexed and so their paths are
// user-controlled: a directory named `proj?v2` is legal on macOS and Linux.
//
// It lives in core because store and embed both need it and store already
// imports embed, so store cannot own it without a cycle.
//
// The "file:" prefix and the escaping are jointly necessary; neither works
// alone, so don't "simplify" one away:
//
//   - Without the prefix, the driver truncates a bare DSN at the first '?' (it
//     splits off what it assumes is a query string), silently opening — and,
//     since it always passes READWRITE|CREATE, *creating* — a database at the
//     truncated path.
//   - Without the escaping, SQLite's own URI parser stops copying the filename
//     at '?' or '#' and percent-decodes '%HH'. That both opens the wrong file
//     and drops any parameters, which is what would turn a "mode=ro"
//     inspection back into one that can create a database.
//
// Parameters are joined verbatim: a caller passing a value that needs escaping
// must escape it.
func SQLiteDSN(path string, params ...string) string {
	esc := dsnPathEscaper.Replace(filepath.ToSlash(path))
	// A leading "//" would be parsed as a URI authority, which SQLite requires
	// to be empty or "localhost". Windows UNC paths (\\server\share) survive
	// filepath.Join, so give them the explicit empty authority they need.
	if strings.HasPrefix(esc, "//") {
		esc = "//" + esc
	}
	uri := "file:" + esc
	if len(params) > 0 {
		uri += "?" + strings.Join(params, "&")
	}
	return uri
}
