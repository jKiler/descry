// Package skill holds the canonical content of the descry agent skill.
//
// It is the single source of truth for the skill's identity (name and trigger
// description), the SKILL.md body, and the shared guidance sentences that other
// agent-facing surfaces (the MCP tool descriptions) reuse. The genskill tool
// renders Markdown() to the committed skills/descry/SKILL.md (drift-checked by
// TestCommittedSkillMatchesGenerator and `make skill-check`), and
// `descry skill install` installs the same rendering into the user-level agent
// skill directories under the user's home.
package skill

import "strings"

// Name is the skill directory name and frontmatter name. It must match the
// installed directory so agents expose it as the /descry command.
const Name = "descry"

// SearchWhen is the canonical "when to reach for descry" sentence. It is
// rendered into the skill body AND the MCP search tool description (the surface
// an agent actually reads at the moment it picks a tool), so the two can never
// drift. Framed around what descry is good at, not around other tools.
const SearchWhen = "Use it whenever you need to locate code: where a feature is implemented, " +
	"which file owns a concept, how something works, or code matching a natural-language " +
	"description or error message."

// Description is the trigger-shaped frontmatter description: what the skill
// does and when to load it. It is the single most important field for the
// agent's decision to use the skill, so it leads with outcomes and keywords.
const Description = "Fast hybrid (semantic + keyword) code search over the current repository, " +
	"backed by the local descry index. " + SearchWhen + " " +
	"One call returns the most relevant files, ranked; for Go it also answers " +
	"call-graph questions (who calls this, is there a path from A to B)."

// Markdown returns the complete SKILL.md document (YAML frontmatter plus body).
// The output is deterministic so it can be regenerated and diff-checked; the
// committed public copy and the user-level installed copies are identical
// renderings of this one function.
func Markdown() string {
	var b strings.Builder
	b.WriteString("---\n")
	b.WriteString("name: " + Name + "\n")
	b.WriteString("description: " + Description + "\n")
	b.WriteString("user-invocable: true\n")
	b.WriteString("---\n")
	b.WriteString(body)
	return b.String()
}

// body is the Markdown instructions an agent reads when the skill activates.
// Keep it focused: how to call descry, how to write queries, and how to read
// the results. Do not embed live state here — the skill is static.
const body = `
# descry

descry is a local hybrid (vector + BM25) code-search index with per-repository
persistence. ` + SearchWhen + `

## How to call it

Preferred — the descry MCP tools, if the ` + "`descry`" + ` MCP server is connected:

- ` + "`search`" + ` — ranked file locations for a query.
- ` + "`read_relevant`" + ` — search plus each match's source content inline, saving a
  follow-up file read.
- ` + "`graph_impact`" + ` — transitive callers of a Go symbol (the blast radius of
  changing it).
- ` + "`graph_trace`" + ` — a call path between two Go symbols, or that none exists.
- ` + "`status`" + ` — which repository is served and every open repository's index
  state. Call it first when results look like the wrong project.

CLI fallback when MCP is not available:

` + "```sh" + `
descry search "<query>"    # run from the repository root
` + "```" + `

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
- If several projects are open, pass ` + "`root`" + ` (an absolute path or just a
  directory name like "kubernetes"); ` + "`status`" + ` lists the candidates.

## If the results are weak

- Rephrase toward the codebase's own vocabulary, or add one distinctive term.
- Ask a narrower question per query rather than several at once.
- descry ranks by relevance; for exhaustive or exact-string sweeps (every caller
  of a symbol outside Go, every occurrence of a literal), use your usual file
  tools on the files descry surfaced.
`
