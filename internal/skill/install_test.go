package skill

import (
	"os"
	"path/filepath"
	"testing"
)

func TestInstallWritesEveryBase(t *testing.T) {
	root := t.TempDir()
	written, err := Install(root)
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if len(written) != len(InstallBases) {
		t.Fatalf("wrote %d paths, want %d", len(written), len(InstallBases))
	}
	for _, base := range InstallBases {
		path := filepath.Join(root, base, Name, "SKILL.md")
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if string(got) != Markdown() {
			t.Errorf("%s content differs from Markdown()", path)
		}
	}
}

func TestInstallIsIdempotentAndRefreshesStaleCopies(t *testing.T) {
	root := t.TempDir()
	if _, err := Install(root); err != nil {
		t.Fatalf("first Install: %v", err)
	}
	// Simulate a stale copy from an older version.
	stalePath := filepath.Join(root, InstallBases[0], Name, "SKILL.md")
	if err := os.WriteFile(stalePath, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Install(root); err != nil {
		t.Fatalf("second Install: %v", err)
	}
	got, err := os.ReadFile(stalePath)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != Markdown() {
		t.Error("re-install did not refresh the stale copy")
	}
}

// A dangling symlink consolidating the two bases (`.claude/skills` →
// `.agents/skills` before either exists) must not break the install, and both
// logical paths must read the skill afterward.
func TestInstallFollowsDanglingBaseSymlink(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".claude"), 0o755); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(root, ".agents", "skills") // does not exist yet
	if err := os.Symlink(target, filepath.Join(root, ".claude", "skills")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	if _, err := Install(root); err != nil {
		t.Fatalf("Install through dangling symlink: %v", err)
	}
	for _, base := range InstallBases {
		path := filepath.Join(root, base, Name, "SKILL.md")
		if _, err := os.ReadFile(path); err != nil {
			t.Errorf("%s not readable after install: %v", path, err)
		}
	}
}

func TestInstallDetectsSymlinkCycle(t *testing.T) {
	root := t.TempDir()
	a := filepath.Join(root, ".claude")
	b := filepath.Join(root, ".agents")
	if err := os.Symlink(b, a); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if err := os.Symlink(a, b); err != nil {
		t.Fatal(err)
	}
	if _, err := Install(root); err == nil {
		t.Fatal("Install must fail on a symlink cycle, not loop")
	}
}

func TestUserStateReflectsInstallAndStaleness(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	if installed, _ := UserState(); installed {
		t.Fatal("fresh home must read as not installed")
	}
	if _, err := InstallUser(); err != nil {
		t.Fatalf("InstallUser: %v", err)
	}
	installed, stale := UserState()
	if !installed || stale {
		t.Fatalf("after install: installed=%v stale=%v, want true/false", installed, stale)
	}
	stalePath := filepath.Join(home, InstallBases[1], Name, "SKILL.md")
	if err := os.WriteFile(stalePath, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	installed, stale = UserState()
	if !installed || !stale {
		t.Fatalf("after corrupting one copy: installed=%v stale=%v, want true/true", installed, stale)
	}
}
