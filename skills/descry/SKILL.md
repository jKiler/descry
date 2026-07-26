---
name: descry
description: Fast hybrid (semantic + keyword) code search over the current repository, backed by the local descry index. Use it whenever you need to locate code: where a feature is implemented, which file owns a concept, how something works, or code matching a natural-language description or error message. One call returns the most relevant files, ranked; for Go it also answers call-graph questions (who calls this, is there a path from A to B).
user-invocable: true
---

# descry

descry is a local hybrid (vector + BM25) code-search index with per-repository
persistence. Use it whenever you need to locate code: where a feature is implemented, which file owns a concept, how something works, or code matching a natural-language description or error message.

## How to call it

Preferred — the descry MCP tools, if the `descry` MCP server is connected:

- `search` — ranked file locations for a query.
- `read_relevant` — search plus each match's source content inline, saving a
  follow-up file read.
- `graph_impact` — transitive callers of a Go symbol (the blast radius of
  changing it).
- `graph_trace` — a call path between two Go symbols, or that none exists.
- `status` — which repository is served and every open repository's index
  state. Call it first when results look like the wrong project.

CLI fallback when MCP is not available:

```sh
descry search "<query>"    # run from the repository root
```

A repository indexes itself on first use. A cold large repository takes minutes;
MCP calls never block on it — they return progress and ask you to retry in a few
seconds. Do exactly that instead of giving up on the tool.

## Writing queries

- Natural language works: describe the behavior ("where are stale runs recovered
  on startup"), name the concept, or paste an error message.
- Distinctive identifiers help: file, package, and symbol names are part of the
  ranking signal, so include them when you know them.
- Results are one per file (its best-matching chunk), best first. Scores are
  comparable only within a single query.
- If several projects are open, pass `root` (an absolute path or just a
  directory name like "kubernetes"); `status` lists the candidates.

## If the results are weak

- Rephrase toward the codebase's own vocabulary, or add one distinctive term.
- Ask a narrower question per query rather than several at once.
- descry ranks by relevance; for exhaustive or exact-string sweeps (every caller
  of a symbol outside Go, every occurrence of a literal), use your usual file
  tools on the files descry surfaced.
