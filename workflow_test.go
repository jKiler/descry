// Contract tests for the GitHub workflows, in the spirit of "anything that can
// silently rot gets a Go test next to it". They pin the load-bearing choices;
// incidental YAML (runner images, action patch versions) is free to change.
package descry_test

import (
	"os"
	"strings"
	"testing"
)

func readWorkflow(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(".github/workflows/" + name)
	if err != nil {
		t.Fatalf("reading workflow: %v", err)
	}
	return string(b)
}

// TestCIWorkflowRunsMakeCheck pins that CI's definition of "passing" is `make
// check` — the Makefile owns the gate (fmt-check, vet, skill-check, tests),
// the workflow only provides runners. Adding steps here instead of to `check`
// would let local `make check` and CI drift apart.
func TestCIWorkflowRunsMakeCheck(t *testing.T) {
	ci := readWorkflow(t, "ci.yml")
	for _, want := range []string{
		"make check",
		"pull_request",            // PRs are gated, not just main
		"go-version-file: go.mod", // go.mod owns the Go version; CI must not pin its own
	} {
		if !strings.Contains(ci, want) {
			t.Errorf("ci.yml no longer contains %q", want)
		}
	}
}

// TestReleaseWorkflowPlatforms pins the released platform set to what descry
// can self-provision ONNX Runtime for (ortAssets in internal/embed/ortlib.go),
// minus windows. In particular darwin/amd64 must NOT be released while the
// pinned ONNX Runtime version publishes no macOS x86_64 archive — the binary
// would build fine and then fail on first use on an Intel mac.
func TestReleaseWorkflowPlatforms(t *testing.T) {
	rel := readWorkflow(t, "release.yml")
	for _, want := range []string{
		"{ goos: linux, goarch: amd64",
		"{ goos: linux, goarch: arm64",
		"{ goos: darwin, goarch: arm64",
	} {
		if !strings.Contains(rel, want) {
			t.Errorf("release.yml no longer builds %q", want)
		}
	}
	if strings.Contains(rel, "goos: darwin, goarch: amd64") {
		t.Error("release.yml builds darwin/amd64, which cannot self-provision ONNX Runtime (see ortAssets); drop it or add a darwin/amd64 ortAssets entry first")
	}
}

// TestReleaseWorkflowContract pins the rest of the release shape: gated on
// `make check`, version stamped from the tag into main.version (which is a var
// for exactly this reason), checksums published, and write permission granted
// only to the job that creates the release, not the whole workflow.
func TestReleaseWorkflowContract(t *testing.T) {
	rel := readWorkflow(t, "release.yml")
	for _, want := range []string{
		"make check",
		"-X main.version=",
		"checksums.txt",
		"tags: [\"v*\"]",
	} {
		if !strings.Contains(rel, want) {
			t.Errorf("release.yml no longer contains %q", want)
		}
	}
	if !strings.Contains(rel, "permissions:\n  contents: read") {
		t.Error("release.yml should default the workflow to contents: read, elevating only the release job")
	}
}
