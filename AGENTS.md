# AGENTS.md

This file is for agentic coding tools working in this repo.

descry is a local hybrid (vector + BM25) code-search index, written in Go. The
CLI entrypoint is `cmd/descry`; implementation lives under `internal/`, and the
package names are the layout map (chunking in `internal/chunk`, language packs
in `internal/lang`, embedding in `internal/embed`, storage in `internal/store`,
retrieval in `internal/search`, the indexing pipeline in `internal/index`, the
call graph in `internal/graph`, the MCP server in `internal/mcp`, shared types
in `internal/core`). `README.md` owns usage, the config env-var table,
and the design notes (retrieval geometry, fingerprint mechanism, measured
quality, and the pipeline provenance table).

Retrieval quality is measured **outside this repository**, by
[codesearch-bench](https://github.com/jKiler/codesearch-bench), which scores
retrieval tools against each other and knows nothing about descry. descry is one
of the tools it drives — as an ordinary subprocess, through `descry index` and
`descry search`.

**Find code with descry itself.** Use the descry MCP tools (`search`,
`read_relevant`, `graph_impact`, `graph_trace`) or `descry search "<query>"` to
locate code by meaning or description — this repo is its own best test corpus.

Verification after non-trivial changes: `make check` (fmt-check, vet,
skill-check, tests). Building needs cgo but not a local ONNX Runtime — the
library and model auto-provision on first use.

**Invariants**

- Bump `pipelineVersion` (`cmd/descry/main.go`) whenever a change improves the
  *quality* of stored data, and add a row to the provenance table in README's
  "How it works" section; `SchemaVersion`
  (`internal/store/sqlite.go`) covers DB layout changes. Embedder/chunker
  identity is fingerprinted automatically.
- `skills/descry/SKILL.md` is **generated**: the source of truth is
  `internal/skill/skill.go`. Edit the source, run `make skill`, commit both;
  `make skill-check` and `TestCommittedSkillMatchesGenerator` fail on drift.
  Never edit the generated file directly.
- Agent-facing guidance is a multi-surface contract: the skill body and the MCP
  `search` tool description share `skill.SearchWhen`, pinned by
  `TestSearchGuidanceSharedWithSkill` (`internal/mcp/guidance_test.go`). Change
  guidance in `internal/skill` and let the surfaces render it; don't fork the
  wording per surface.
- The CLI is verb-first: any argument that is not a reserved subcommand is a
  search query (`parseArgs` in `cmd/descry/main.go`). Adding a subcommand name
  is therefore a breaking change for one-word queries — extend the
  `subcommands` set deliberately, and never prompt when stdin is not a TTY.
- Every sqlite open goes through `core.SQLiteDSN` (index, embed cache,
  diagnostics). These databases sit inside the repository being indexed, so
  their paths are user-controlled, and a bare DSN is silently truncated at the
  first `?` — creating a stray database at the truncated path. The `file:`
  prefix and the path escaping are jointly necessary; the doc comment explains
  why, and `TestInspectHandlesURISyntaxInPath` /
  `TestCachedEmbedderHandlesURISyntaxInPath` pin it. It lives in `core` because
  `store` already imports `embed`, so `store` can't own it without a cycle.
- Diagnostics must be read-only, and must not fetch. `descry doctor` reports
  the resolution chain via the `Locate*` probes (`embed.LocateOnnxRuntime`,
  `embed.LocateModel`) and `store.Inspect`, never the `Ensure*` functions or
  `store.OpenSQLite` — opening an index whose fingerprint has moved on *clears
  it*, so a diagnostic built on it would destroy what it was asked about.
  Pinned by `TestInspectDoesNotClearAStaleIndex` and
  `TestLocateModelDoesNotDownload`. Each `Ensure*` is defined in terms of its
  `Locate*` so the two can't describe different chains.
- **descry does not benchmark itself, and has no in-repo harness.**
  `$DESCRY_CHUNKER` selects the chunking strategy and `descry index -json`
  reports the file and chunk counts a harness charges it for, so an external
  benchmark can run two arms that differ in exactly one thing. That is the rule
  for descry's CLI generally: anything a comparison needs to vary has to be
  reachable from it, because a knob only internal code can turn is a measurement
  nobody else can reproduce. `TSChunker.ID()` derives from its settings and
  feeds the index fingerprint, so a non-default configuration rebuilds rather
  than reusing the default's chunks.
- **Adding a language is adding a Pack** (`internal/lang`), never editing the
  chunker. A pack is data: extensions, test/generated-file conventions, a
  tree-sitter grammar, and a query whose captures (`@chunk`, `@scope`, `@drop`,
  `@name`, `@qualifier`, `@body`) are the entire interface to
  `chunk.TSChunker`. If a language cannot be expressed in that vocabulary,
  extend the vocabulary — a branch on a language's name in the core chunker is
  the bug this design exists to prevent.
- A tree-sitter query that fails to compile does not fail loudly: it silently
  degrades its whole language to the fallback chunker, for every file. That has
  already happened once (TypeScript, over `identifier` vs `type_identifier`).
  `TestPackQueriesCompile` is the guard, and is the first thing to check when
  one language's results collapse.
- Chunking is a **cover**: every byte of a parsed file lands in some chunk, with
  uncovered regions becoming gap chunks attributed to their enclosing scope.
  This is what makes "never silently vanish from the index" a property of the
  algorithm rather than of how carefully each query was written. The
  `coversEveryLine` helper in `internal/chunk/ts_test.go` pins it.
- Retrieval defaults (`RRFK`, `VecWeight`, `LexWeight`, `LexFileWeight`,
  `FuseAlpha`, `CandMult` in `search.NewHybrid`) are the kubernetes-120 sweep
  optimum. Don't change them without re-measuring — `descry eval` on an external
  corpus, or codesearch-bench — and keep README's "Measured quality" table in
  step with any retrieval change.
- **Chunk *selection* — which chunk of a file `SearchFiles` shows — is a closed
  question.** A symbol ranker, a symbol field inside BM25F, and a cross-encoder
  were each built and measured; all three were reverted, the cross-encoder (the
  standard answer) worst of all. They fail for one structural reason: every
  available signal scores how much a chunk *resembles* the query, and the
  siblings being chosen between resemble it equally. Reopening it needs a signal
  that is not resemblance. What is still open is *file* recall, and how many
  chunks a result carries — which is what `chunks_per_file` exists for.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session here.
Do not repeat what the codebase already shows; point to the authoritative file
instead. Prefer rewriting or pruning existing entries over appending new ones.
