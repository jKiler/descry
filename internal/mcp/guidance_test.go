package mcp

import (
	"strings"
	"testing"

	"github.com/jKiler/descry/internal/skill"
)

// Agent-facing guidance is a multi-surface contract: the skill body (loaded
// when the skill triggers) and the MCP search tool description (read at the
// moment an agent picks a tool) must carry the same "when to reach for descry"
// guidance. Both render skill.SearchWhen by construction; this test guards
// against either surface being unwired from the shared constant, and pins the
// load-bearing phrases so the constant itself can't quietly lose them.
func TestSearchGuidanceSharedWithSkill(t *testing.T) {
	surfaces := map[string]string{
		"mcp search tool description": searchToolDescription,
		"skill markdown":              skill.Markdown(),
	}
	for name, content := range surfaces {
		if !strings.Contains(content, skill.SearchWhen) {
			t.Errorf("%s no longer renders skill.SearchWhen", name)
		}
		for _, phrase := range []string{
			"locate code",
			"natural-language",
		} {
			if !strings.Contains(content, phrase) {
				t.Errorf("%s is missing the canonical phrase %q", name, phrase)
			}
		}
	}
}
