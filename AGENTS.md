# AGENTS.md

This file is for agentic coding tools working in this repo.

descry is a local hybrid (vector + BM25) code-search index, written in Go. The
CLI entrypoint is `cmd/descry`; implementation lives under `internal/`, and the
package names are the layout map (chunking in `internal/chunk`, embedding in
`internal/embed`, storage in `internal/store`, retrieval in `internal/search`,
the indexing pipeline in `internal/index`, the call graph in `internal/graph`,
the MCP server in `internal/mcp`, shared types in `internal/core`).
`README.md` owns usage, the config env-var table, and the design notes
(retrieval geometry, fingerprint mechanism, benchmarks, and the pipeline
provenance table).

**Find code with descry itself.** Use the descry MCP tools (`search`,
`read_relevant`, `graph_impact`, `graph_trace`) or `descry search "<query>"` to
locate code by meaning or description — this repo is its own best test corpus.

Verification after non-trivial changes: `make check` (fmt-check, vet,
skill-check, tests). Building needs cgo but not a local ONNX Runtime — the
library and model auto-provision on first use.

**Invariants**

- Bump `pipelineVersion` (`cmd/descry/main.go`) whenever a change improves the
  *quality* of stored data, and add a row to the provenance table in README's
  "Reindexing & the fingerprint" section; `SchemaVersion`
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
- Retrieval defaults (`RRFK`, `VecWeight`, `LexWeight`, `LexFileWeight`,
  `FuseAlpha`, `CandMult` in `search.NewHybrid`) are the kubernetes-120 sweep
  optimum. Don't change them without re-running `descry eval` on an external
  corpus, and keep README's "Measured quality" table in step with any
  retrieval change.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session here.
Do not repeat what the codebase already shows; point to the authoritative file
instead. Prefer rewriting or pruning existing entries over appending new ones.
