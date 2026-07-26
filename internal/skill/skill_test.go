package skill

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMarkdownShape(t *testing.T) {
	md := Markdown()

	if !strings.HasPrefix(md, "---\nname: "+Name+"\n") {
		t.Errorf("frontmatter must start with the skill name; got prefix %q", md[:40])
	}
	// The description is a YAML scalar on one line; an embedded newline would
	// break the frontmatter of every installed copy.
	if strings.Contains(Description, "\n") {
		t.Error("Description must be a single line")
	}
	if !strings.Contains(md, "description: "+Description+"\n") {
		t.Error("frontmatter must carry the exact Description constant")
	}

	// The body must teach every MCP tool and the CLI fallback.
	for _, want := range []string{
		"`search`", "`read_relevant`", "`graph_impact`", "`graph_trace`", "`status`",
		"descry \"<query>\"",
		"descry index",
		SearchWhen,
	} {
		if !strings.Contains(md, want) {
			t.Errorf("skill body is missing %q", want)
		}
	}

	if md != Markdown() {
		t.Error("Markdown must be deterministic")
	}
}

// TestCommittedSkillMatchesGenerator is the drift gate: the committed
// skills/descry/SKILL.md must be exactly what the generator renders. Run
// `make skill` (go run ./cmd/genskill) after editing internal/skill and commit
// the result.
func TestCommittedSkillMatchesGenerator(t *testing.T) {
	path := filepath.Join("..", "..", "skills", Name, "SKILL.md")
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read committed skill: %v (run `make skill` and commit the result)", err)
	}
	if string(got) != Markdown() {
		t.Fatalf("%s is stale; run `make skill` and commit the result", path)
	}
}
