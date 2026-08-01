# descry

[![ci](https://github.com/jKiler/descry/actions/workflows/ci.yml/badge.svg)](https://github.com/jKiler/descry/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/jKiler/descry)](https://github.com/jKiler/descry/releases)

> **descry** *(v.)* — to catch sight of something distant or difficult to discern.

A local, hybrid code-search index for AI coding agents and humans, written in Go.

descry walks a repository, splits each file into semantic chunks, embeds them with
a sentence-transformer model, and serves **hybrid retrieval** — dense vector
search fused with BM25 lexical search — over a persistent SQLite index. It also
builds a Go call/dependency graph for impact and trace queries.

![descry cold start: it asks before indexing, embeds the repo with live progress, answers a natural-language query with ranked file:line results, then reports the whole resolution chain with descry doctor](demo.gif)

*Recorded with the embedding model already cached. A true first run downloads
it (~90MB) once before indexing.*

- **Hybrid retrieval** — MiniLM vectors + BM25 (chunk-level and whole-file),
  fused by rank. Semantic recall plus exact-identifier precision.
- **Syntax-aware chunking** — Go, Python, TypeScript/TSX, JavaScript, Rust,
  Java, C and C++ split on real declaration boundaries, doc comments attached,
  oversized declarations windowed so a long function's tail still gets embedded.
  Each language is a data-only pack; anything else falls back to paragraphs.
  Symbols are qualified syntactically (receiver, class, `impl`), not by type
  resolution — outside Go that would mean a compiler per language.
- **Fast at scale** — 275k chunks (the Kubernetes tree) answer in ~67ms once
  warm, after a ~3.8s load. Exact, not approximate.
- **Self-healing index** — a fingerprint over schema, pipeline, embedder and
  chunker; any mismatch rebuilds transparently, reusing a persistent vector cache.
- **Call graph** — name-based or `go/types`-checked, rendered as Mermaid.
- **MCP server** — `search`, `read_relevant`, `graph_impact`, `graph_trace`,
  `status` over stdio.
- **`descry eval`** — Recall@k and MRR for your own labeled query set.

No external services. The model downloads once from Hugging Face and is cached.

## Install

```bash
go install github.com/jKiler/descry/cmd/descry@latest
```

Or a prebuilt binary from the [releases page](https://github.com/jKiler/descry/releases).

Building needs a C compiler (cgo). ONNX Runtime is *dlopen*ed, not linked — it
downloads and caches on first use, sha256-verified. Point `DESCRY_ORT_LIB` at
your own copy to skip that.

## Usage

```bash
descry "where is auth handled"     # any non-subcommand argument is a query
descry                             # index status, or offer to build one
descry index .                     # explicit forms, for scripts
descry search "where is auth handled"
descry graph . > graph.mmd         # Go call graph as Mermaid
descry eval eval_queries.json      # score against a labeled set
descry mcp                         # serve to an agent over MCP
descry skill install               # teach coding agents to use descry
descry doctor                      # health-check the whole resolution chain
```

A query against an unindexed repository asks before indexing, and never prompts
when stdin isn't a TTY.

`descry doctor` is read-only — it never downloads and never rebuilds — and exits
1 on anything broken, so CI can gate on it:

```
✓ build        descry 0.3.0  darwin/arm64  go1.26.2  cgo enabled
✓ onnxruntime  /opt/homebrew/lib/libonnxruntime.dylib (system install)
✓ model        all-MiniLM-L6-v2 (cached, 86.2 MB)
– index        483 chunks in .descry/index.db — stale: embedder all-MiniLM-L6-v2 → …-q8
               the next index or search rebuilds it automatically
– agent skill  not installed
```

### MCP

```json
{ "mcpServers": { "descry": { "command": "descry", "args": ["mcp"] } } }
```

One global server, no per-project setup: the repository is resolved per call
from the call's `root`, then an explicit path or `DESCRY_ROOT`, then the
client's MCP roots, then the working directory. Every result names the
repository it came from.

`search` and `read_relevant` return **two spans per file** by default — the
common failure is not the wrong file but the wrong part of the right one. A
second span buys a mean +9.6pp chunk recall for 1.86× the tokens;
`chunks_per_file` (1–3) is the context-budget dial, and `k` counts files.

Querying an unindexed repository starts the build in the background and returns
progress rather than blocking.

### Configuration

| Env var | Meaning | Default |
| --- | --- | --- |
| `DESCRY_VEC_WEIGHT` | fusion weight of the vector ranking | 1.1 |
| `DESCRY_LEX_WEIGHT` | fusion weight of the chunk-level BM25 ranking | 0.5 |
| `DESCRY_LEXFILE_WEIGHT` | fusion weight of the whole-file BM25 ranking | 1.0 |
| `DESCRY_RRF_K` | RRF constant (larger flattens top ranks) | 25 |
| `DESCRY_FUSE_ALPHA` | consensus dial: 1 = plain RRF sum, 0 = best list only | 0.40 |
| `DESCRY_CAND_MULT` | candidate depth per ranker, as a multiple of topK | 8 |
| `DESCRY_BM25_PATH` | set to `0` to drop path/symbol tokens from BM25 | on |
| `DESCRY_RERANK_MULT` | candidates kept from the int8 scan for float32 rerank | 6 |
| `DESCRY_MODEL` | `q8` for the int8 model (faster indexing) | fp32 |
| `DESCRY_ORT_LIB` | your own onnxruntime shared library | auto-provisioned |
| `DESCRY_INDEX_DIR` | keep the index outside the repository, keyed per root | `<repo>/.descry` |
| `DESCRY_CHUNKER` | `ts`, `ast-go` or `line`; changes the fingerprint, so switching rebuilds | `ts` |

These are runtime overrides — retune without reindexing.

## How it works

```
walk → chunk → embed → store            (index)
query → embed ┐
              ├─ rank fusion → results  (search)
query → BM25 ─┘
```

Three rankings — vector, chunk BM25, whole-file BM25 — each collapsed to files
before fusion. Fusion is rank-based but not plain RRF: `best_list + α·(sum −
best_list)` with α = 0.40, so one strongly-convinced ranker can carry a result
the others missed. BM25 loads a persisted inverted index; vector search scans an
int8 matrix and reranks the top candidates in exact float32.

The index records a fingerprint — schema, pipeline, embedder and dimension,
chunker — and any mismatch clears and rebuilds it on open. Rebuilds are cheap:
vectors live in a separate cache keyed by `(embedder id, sha256 of embed text)`
that survives clears.

`pipelineVersion` (`cmd/descry/main.go`) is bumped whenever a change improves
stored-data *quality*:

| Version | Change |
| ------- | ------ |
| 1 | Baseline: AST chunking, BM25 + RRF, call graph. |
| 2 | Exclude test and generated files from the index. |
| 3 | Embed each chunk as `path\ncontent`. |
| 4 | Tree-sitter chunking for eight languages, enriched embed text, windowed oversized declarations, per-language test/generated detection. |

## Measured quality

**File granularity.** Kubernetes, 120 keyword-dense queries, one gold file each:
R@10 95.8%, MRR 0.726 (fp32) — the shipped weights are that sweep's optimum.

**Chunk granularity**, measured by
[codesearch-bench](https://github.com/jKiler/codesearch-bench) over seven pinned
repositories and 523 queries whose gold is a line span. Recall@20, tree-sitter
against the chunking it replaced, same binary both arms:

| Corpus | before | after | |
| --- | ---: | ---: | ---: |
| django | 34.2% | 48.7% | +14.5 |
| leveldb | 40.5% | 54.1% | +13.6 |
| vue-core | 45.8% | 54.2% | +8.4 |
| kubernetes | 25.3% | 30.7% | +5.4 |
| ripgrep | 39.7% | 41.1% | +1.4 |
| spring-boot | 16.4% | 17.8% | +1.4 |
| redis | 46.2% | 45.0% | **−1.2** |

Six improve, redis regresses. The bigger result is cost: `go/ast` emits one
unbounded chunk per declaration, so on kubernetes it spends **23,570 tokens per
query against tree-sitter's 2,243** — better answers for a tenth of the context.
(Both arms indexed the whole kubernetes checkout, so that row isn't comparable
with figures measured over its subtrees.)

Chunk numbers are far below file numbers by construction, and the gap is the
point: it is how often descry finds the right file and shows the wrong part.

## Layout

```
cmd/descry/         CLI (index / status / search / graph / eval / mcp / skill)
cmd/genskill/       renders skills/descry/SKILL.md from internal/skill
internal/core/      shared domain types
internal/chunk/     file → chunks (TSChunker, ASTChunker, LineChunker)
internal/lang/      language packs: extensions, test rules, grammar + query
internal/tokenize/  identifier splitting
internal/embed/     text → vector (ONNX MiniLM) + cache + model fetch
internal/store/     chunk storage + nearest-neighbor
internal/search/    BM25, RRF fusion, hybrid retriever
internal/graph/     call graph + Mermaid
internal/index/     walk → chunk → embed → store
internal/eval/      Recall@k, MRR
internal/mcp/       MCP server (stdio)
internal/skill/     agent-skill content + installer
```

## Requirements

Go 1.26+ and a C compiler (cgo). ONNX Runtime downloads on first use.

cgo is unavoidable: every tree-sitter grammar binding for Go compiles the
generated `parser.c`, including the collections usually described as pure-Go.

## License

[Apache-2.0](./LICENSE)
