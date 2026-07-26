# descry

[![ci](https://github.com/jKiler/descry/actions/workflows/ci.yml/badge.svg)](https://github.com/jKiler/descry/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/jKiler/descry)](https://github.com/jKiler/descry/releases)

> **descry** *(v.)* — to catch sight of something distant or difficult to discern.

A local, hybrid code-search index for AI coding agents and humans, written in Go.

descry walks a repository, splits each file into semantic chunks, embeds them with
a sentence-transformer model, and serves **hybrid retrieval** — dense vector
search fused with BM25 lexical search — over a persistent SQLite index. It also
builds a Go call/dependency graph for impact and trace queries.

![descry cold start: it asks before indexing, embeds the repo with live progress, answers a natural-language query with ranked file:line results, then answers a second query instantly from the warm index](demo.gif)

- **Hybrid retrieval** — vector (all-MiniLM-L6-v2) + BM25, fused with Reciprocal
  Rank Fusion. Semantic recall plus exact-identifier precision.
- **Indexed and fast at scale** — a persisted BM25 inverted index and an int8
  vector scan with float32 rerank keep a ~96k-chunk repo (Kubernetes) at ~0.5s
  startup and ~25ms queries. Results are exact — no approximation.
- **AST-aware chunking** — Go files are split on real declaration boundaries
  (func / type / method) with doc comments attached; everything else falls back
  to paragraph chunking.
- **Persistent, self-healing index** — chunks and vectors live in a single
  `.descry/index.db`. The index carries a fingerprint and rebuilds itself
  automatically when the model, chunker, or pipeline changes.
- **Fast embedding** — ONNX Runtime (~4.7ms/embed) behind a concurrent worker
  pool and a persistent vector cache, so re-indexing after a change is cheap.
- **Call graph** — name-based or type-checked (`go/types`) dependency graph,
  rendered as Mermaid.
- **MCP server** — exposes `search`, `read_relevant`, `graph_impact`,
  `graph_trace`, and `status` to AI agents over the Model Context Protocol
  (JSON-RPC on stdio).
- **Evaluation harness** — Recall@k and MRR against a labeled query set.

No external services: the model downloads once from Hugging Face on first use and
is cached locally; everything else runs on your machine.

## Install

```bash
go install github.com/jKiler/descry/cmd/descry@latest
```

Or grab a prebuilt binary for macOS (Apple silicon) or Linux (amd64/arm64)
from the [releases page](https://github.com/jKiler/descry/releases).

Embedding runs on ONNX Runtime. The Go binding *dlopens* the library rather than
linking it, so **building needs only a C compiler (cgo)** — the onnxruntime
library itself is downloaded and cached on first use (pinned version, verified by
sha256). Nothing to install by hand.

If you'd rather supply your own library (a distro package, Homebrew, a custom
build), point `DESCRY_ORT_LIB` at it and descry will use that instead. An
existing system install is picked up automatically too.

A `Makefile` wraps the common developer tasks — run `make help` to list them
(`make build`, `make test`, `make check`, …).

## Usage

```bash
# The verb: any argument that isn't a subcommand is a search query.
# On a repo with no index, descry names the directory and asks before indexing
# (the very first run also downloads the ~90MB embedding model, once).
descry "where is auth handled"
descry how is auth implemented        # quotes optional

# Bare `descry` shows this repo's index status (or offers to build it).
descry

# The explicit forms, for scripts: index / status / search without prompts.
descry index .
descry status
descry search "where is auth handled"

# Print the Go call graph as Mermaid (typed by default; --named forces the heuristic).
descry graph . > graph.mmd

# Score retrieval quality against a labeled query set.
descry eval eval_queries.json

# Serve the index to an AI agent over the Model Context Protocol (stdio).
descry mcp

# Install the agent skill so coding agents reach for descry when they search.
descry skill install

# Health-check everything that has to resolve before a query can be answered.
descry doctor
```

### `descry doctor`

First runs have a deep resolution chain — cgo, the ONNX Runtime library, the
model files, the index and its fingerprint — and when it works, all of it is
invisible. `doctor` reports the whole chain at once instead of surfacing it one
error at a time:

```
✓ build        descry 0.1.0  darwin/arm64  go1.26.2  cgo enabled
✓ onnxruntime  /opt/homebrew/lib/libonnxruntime.dylib (system install)
✓ model        all-MiniLM-L6-v2 (cached, 86.2 MB)
– index        483 chunks in .descry/index.db — stale: embedder all-MiniLM-L6-v2 → all-MiniLM-L6-v2-q8
               the next index or search rebuilds it automatically
✓ embed cache  1.0 MB — a rebuild reuses these vectors
– agent skill  not installed
               fix: run `descry skill install` so coding agents search with descry
```

It is strictly read-only: it never downloads (an uncached model reports as
"downloaded on first index" rather than fetching ~90MB from a diagnostic), and
it never rebuilds — it reads the index's recorded fingerprint directly rather
than opening it, because opening an index with a moved-on fingerprint *clears*
it by design. A stale index names the field that drifted, so a surprise rebuild
has a visible cause. `✗` marks something descry cannot work without and exits
1, so scripts and CI can gate on it; `–` is advisory. Any retrieval knobs left
in the environment (`DESCRY_VEC_WEIGHT` and friends) are reported too, since a
stale override silently changes every result with no other symptom.

A query against a cold repository asks for consent (naming the directory) and
then indexes and answers in one run; non-interactive runs never prompt or
silently index — they exit with the `descry index` hint instead. Later runs
reuse the persisted index. Indexing shows a progress bar on stderr.

### MCP

`descry mcp [dir]` speaks the Model Context Protocol over stdio, so an agent can
query your repository through five tools: `search`, `read_relevant` (search +
inline source), `graph_impact`, `graph_trace`, and `status`.

Configure it once, globally, and it follows whichever project you have open:

```json
{
  "mcpServers": {
    "descry": { "command": "descry", "args": ["mcp"] }
  }
}
```

No per-project configuration is needed. The repository is resolved **per call**,
in order:

1. the call's own `root` argument — lets an agent target one of several open
   projects, by path or just by name (`"kubernetes"`);
2. an explicit path (`descry mcp /path/to/repo`, or `DESCRY_ROOT`) — pins the
   server to one repository;
3. the client's **MCP roots** — the project your editor currently has open, so
   one global server serves every project you switch to;
4. the process working directory, as a last resort.

Every result names the repository it came from, so an answer from the wrong
project is obvious rather than silent. If several projects are open, `status`
lists them all with their index state, and `root_source` explains which rule
picked the current one.

**Indexing never blocks a call.** Querying a repository that isn't indexed yet
starts the build in the background and returns progress immediately
(`"Indexing …: 4200/95953 chunks embedded (4%)"`), so even a very large
repository can't stall or time out a request — retry a few seconds later. Run
`descry index` beforehand to have results available straight away.

Repositories are opened lazily and cached per root, so switching projects just
opens another one. The index is written to `<root>/.descry/`, so the root must be
writable — if it isn't, the tool call returns a clear error instead of the
server dying.

### Agent skill

`descry skill install` writes the `/descry` skill into the user-level agent
skill directories (`~/.claude/skills` and `~/.agents/skills`, read by Claude
Code, Codex, OpenCode, and others). The skill teaches agents when to reach for
descry — locating code by meaning, concept, or error message — and how to drive
the MCP tools and the CLI fallback. It's one file, installed once per machine;
re-run the command after upgrading descry to refresh it. The committed copy
lives at `skills/descry/SKILL.md`, generated from `internal/skill` (`make
skill`).

### Configuration

Retrieval weights and depth are overridable at runtime (no rebuild), which makes
tuning cheap — build the index once and re-run `eval` with different values:

| Env var | Meaning | Default |
| --- | --- | --- |
| `DESCRY_VEC_WEIGHT` | fusion weight of the vector ranking | 1.1 |
| `DESCRY_LEX_WEIGHT` | fusion weight of the chunk-level BM25 ranking | 0.5 |
| `DESCRY_LEXFILE_WEIGHT` | fusion weight of the whole-file BM25 ranking | 1.0 |
| `DESCRY_RRF_K` | RRF constant (larger flattens top ranks) | 25 |
| `DESCRY_FUSE_ALPHA` | consensus dial: 1 = plain RRF sum, 0 = best list only | 0.40 |
| `DESCRY_CAND_MULT` | candidate depth per ranker, as a multiple of topK | 8 |
| `DESCRY_BM25_PATH` | set to `0` to drop path/symbol tokens from BM25 | on |
| `DESCRY_RERANK_MULT` | candidates per result kept from the int8 vector scan for exact float32 rerank | 6 |
| `DESCRY_MODEL` | set to `q8` to use the int8-quantized model (faster indexing) | fp32 |
| `DESCRY_ORT_LIB` | path to your own onnxruntime shared library | auto-provisioned |

## How it works

```
walk → chunk → embed → store            (index)
query → embed ┐
              ├─ rank fusion → results  (search)
query → BM25 ─┘
```

Every stage sits behind a small interface (`Chunker`, `Embedder`, `Store`), so
implementations swap without touching callers.

### Retrieval

A query produces **three rankings**, each collapsed to files (a file's rank is
its best chunk's rank) before fusion:

1. **vector** — cosine over MiniLM embeddings;
2. **chunk BM25** — lexical match at declaration granularity, which still
   surfaces one sharply-matching function inside a large file;
3. **whole-file BM25** — a second BM25 whose documents are entire files, so
   term frequency and length normalization operate on the real unit of
   retrieval. The strongest lexical signal.

Fusion is rank-based (scores across rankers aren't comparable) but not plain
RRF: `score = best_list + α · (sum − best_list)` with α = 0.40, so one
strongly-convinced ranker can carry a result that the others missed — plain
summed RRF buries exactly those. Two extra signals help queries that name a
component: each chunk's path and symbol tokens join its BM25 bag, and each
chunk embeds as `path\ncontent` so vectors carry the component name too.

Both halves are indexed rather than scanned: BM25 loads a persisted inverted
index, and vector search scans an int8-quantized matrix then reranks the top
candidates in exact float32 — lossless (verified against exhaustive float
search), at int8 speed. On Kubernetes (~96k chunks) that keeps startup at
~0.5s and a warm query at ~25ms.

### Reindexing & the fingerprint

The index records a fingerprint: schema version, pipeline version, embedder id
and dimension, and chunker id. Any mismatch on open clears and rebuilds the
index transparently — no manual invalidation, no stale results. Rebuilds are
cheap because vectors live in a separate read-through cache keyed by
`(embedder id, sha256 of the embed text)` that survives index clears: only new
or edited chunks touch the model.

`pipelineVersion` (in `cmd/descry/main.go`) is bumped whenever a change
improves the *quality* of stored data. Provenance of each bump:

| Version | Change |
| ------- | ------ |
| 1 | Baseline: AST chunking, BM25 + RRF, call graph. |
| 2 | Exclude `_test.go` and generated files from the index. |
| 3 | Embed each chunk as `path\ncontent` so vectors carry the component name. |

### Embedding

**all-MiniLM-L6-v2** (384-dim), the standard sentence-transformers recipe:
WordPiece tokenize → ONNX forward pass → mask-weighted mean pooling → L2
normalization. Inference runs on ONNX Runtime (~4.7ms per chunk) across a
`GOMAXPROCS` worker pool; `DESCRY_MODEL=q8` swaps in the int8 model for ~1.9×
faster indexing at a sub-point quality cost. Query latency is retrieval-bound,
not model-bound — a query embeds one short string, cached after first use.

### Measured quality

Kubernetes corpus (~96k chunks after excluding tests and generated code), 120
keyword-dense queries, gold = one file each, everything at shipped defaults:

| Model | R@5 | R@10 | R@20 | MRR |
| --- | --- | --- | --- | --- |
| q8 | 88.3% | 95.8% | 98.3% | 0.720 |
| fp32 | 89.2% | 95.8% | 98.3% | 0.726 |

A 66-query natural-language set over descry's own source reads R@10 ~97%. The
shipped weights are the optimum of a sweep on the Kubernetes set; `descry eval`
re-scores any labeled query set against the current configuration.

## Layout

```
cmd/descry/         CLI (index / status / search / graph / eval / mcp / skill)
cmd/genskill/       renders skills/descry/SKILL.md from internal/skill
skills/descry/      the committed agent skill (generated — edit internal/skill)
internal/core/      shared domain types (Chunk, SearchResult)
internal/chunk/     file → chunks (LineChunker, ASTChunker)
internal/tokenize/  camelCase / snake_case identifier splitting
internal/embed/     text → vector (ONNX MiniLM, HashEmbedder) + cache + model/runtime fetch
internal/store/     chunk storage + nearest-neighbor (MemStore, SQLiteStore)
internal/search/    BM25, RRF fusion, hybrid retriever
internal/graph/     dependency/call graph (name-based and typed) + Mermaid
internal/index/     the walk → chunk → embed → store orchestrator
internal/eval/      Recall@k, MRR, evaluation harness
internal/mcp/       Model Context Protocol server (stdio) exposing the tools
internal/skill/     canonical agent-skill content + user-level installer
```

## Requirements

- Go 1.26+ (per `go.mod`) and a C compiler (cgo) to build.
- The onnxruntime shared library is downloaded on first use; no manual install.

## License

[Apache-2.0](./LICENSE)
