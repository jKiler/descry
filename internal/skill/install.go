package skill

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// InstallBases are the user-level agent skill parent directories, relative to
// the user's home directory. `~/.claude/skills` is Claude Code's personal-skill
// location (OpenCode reads it too); `~/.agents/skills` is the vendor-neutral
// user-level convention Codex, OpenCode, and others read.
var InstallBases = []string{
	filepath.Join(".claude", "skills"),
	filepath.Join(".agents", "skills"),
}

// InstallUser installs the skill into the agent skill directories under the
// current user's home directory. It returns the home-relative paths written.
func InstallUser() ([]string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("resolve home directory: %w", err)
	}
	return Install(home)
}

// UserState reports whether the skill is installed under the current user's
// home (any base has a SKILL.md) and whether any installed copy is stale
// (differs from the current rendering). Best-effort: an unreadable home reads
// as not installed.
func UserState() (installed, stale bool) {
	home, err := os.UserHomeDir()
	if err != nil {
		return false, false
	}
	want := Markdown()
	for _, base := range InstallBases {
		got, err := os.ReadFile(filepath.Join(home, base, Name, "SKILL.md"))
		if err != nil {
			continue
		}
		installed = true
		if string(got) != want {
			stale = true
		}
	}
	return installed, stale
}

// Install writes SKILL.md into each agent skills directory under root
// (normally the user's home directory), creating directories as needed. It
// returns the root-relative paths written. Writing is idempotent: re-running
// overwrites with identical content, which also refreshes a stale copy left by
// an older descry version.
//
// Users may consolidate the two bases with a symlink — `.claude/skills` →
// `.agents/skills`, the whole `.claude` dir → `.agents`, or the reverse.
// Install follows such links transparently, including when the link's target
// does not exist yet (a plain os.MkdirAll would fail with "file exists" on a
// dangling symlink). Both logical bases stay readable afterward via the link.
func Install(root string) ([]string, error) {
	content := []byte(Markdown())
	written := make([]string, 0, len(InstallBases))
	for _, base := range InstallBases {
		rel := filepath.Join(base, Name, "SKILL.md")
		dir := filepath.Join(root, base, Name)
		realDir, err := resolveThroughSymlinks(dir)
		if err != nil {
			return written, err
		}
		if err := os.MkdirAll(realDir, 0o755); err != nil {
			return written, err
		}
		if err := os.WriteFile(filepath.Join(realDir, "SKILL.md"), content, 0o644); err != nil {
			return written, err
		}
		written = append(written, rel)
	}
	return written, nil
}

// maxSymlinkHops bounds link expansion during resolution. A genuine cycle
// exhausts it; a legitimate path may cross the same link more than once (on
// macOS every temp path passes through /var → /private/var, and a base symlink
// whose target is spelled via /var crosses it again), so a hop budget is the
// cycle guard — not a visited set, which would flag that second crossing.
const maxSymlinkHops = 40

// resolveThroughSymlinks walks dir component by component and rewrites the path
// through any symlink it encounters, even one whose target does not exist yet.
// The result contains no symlink components, so os.MkdirAll on it cannot trip
// over a dangling symlink. dir must be absolute.
func resolveThroughSymlinks(dir string) (string, error) {
	hops := 0
	return resolveHops(dir, &hops)
}

func resolveHops(dir string, hops *int) (string, error) {
	clean := filepath.Clean(dir)
	volume := filepath.VolumeName(clean)
	cur := volume + string(filepath.Separator)
	for _, part := range strings.Split(strings.TrimPrefix(clean, volume), string(filepath.Separator)) {
		if part == "" {
			continue
		}
		cur = filepath.Join(cur, part)
		info, err := os.Lstat(cur)
		if err != nil {
			// This component does not exist yet; nothing left to resolve —
			// the remaining parts are appended verbatim onto the prefix.
			continue
		}
		if info.Mode()&os.ModeSymlink == 0 {
			continue
		}
		*hops++
		if *hops > maxSymlinkHops {
			return "", fmt.Errorf("too many symlinks resolving %s", dir)
		}
		target, err := os.Readlink(cur)
		if err != nil {
			return "", err
		}
		if !filepath.IsAbs(target) {
			target = filepath.Join(filepath.Dir(cur), target)
		}
		// The target may itself be or contain symlinks; resolve recursively.
		if cur, err = resolveHops(target, hops); err != nil {
			return "", err
		}
	}
	return cur, nil
}
