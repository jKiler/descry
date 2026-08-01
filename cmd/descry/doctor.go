// `descry doctor` — a read-only health report for everything that has to
// resolve before a query can be answered: the build, the ONNX Runtime library,
// the model files, the index and its fingerprint, the embed cache, the agent
// skill, and any retrieval tuning left in the environment.
//
// Two rules shape this file. It never *fixes* anything, and it never fetches:
// each check reports where a thing resolves from and, on a miss, the exact
// command or variable that fixes it. That is why the checks call the Locate*
// probes rather than the Ensure* functions, and store.Inspect rather than
// store.OpenSQLite (which clears a stale index by design).
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/jKiler/descry/internal/embed"
	"github.com/jKiler/descry/internal/skill"
	"github.com/jKiler/descry/internal/store"
)

// mark is a check's verdict: healthy, advisory, or broken.
type mark int

const (
	markOK   mark = iota // ✓ resolved and usable
	markInfo             // – not an error: absent-but-optional, or work deferred to first use
	markBad              // ✗ descry cannot work until this is fixed
)

func (m mark) String() string {
	switch m {
	case markOK:
		return "✓"
	case markInfo:
		return "–"
	case markBad:
		return "✗"
	}
	// Never reached; a new mark should be visible as unrendered rather than
	// silently reported as fatal (the exit code keys off markBad, not glyphs).
	return "?"
}

// check is one reported line. hint is the second line: a fix on a miss, or the
// reason a "–" is harmless.
type check struct {
	mark   mark
	label  string
	detail string
	hint   string
}

// runDoctor prints the health report for dir and exits 1 if any check is
// broken, so scripts and CI can gate on `descry doctor` succeeding.
func runDoctor(dir string) {
	checks := doctorChecks(dir)

	width := 0
	for _, c := range checks {
		width = max(width, len(c.label))
	}
	for _, c := range checks {
		fmt.Printf("%s %-*s  %s\n", c.mark, width, c.label, c.detail)
		if c.hint != "" {
			fmt.Printf("  %-*s  %s\n", width, "", c.hint)
		}
	}
	if slices.ContainsFunc(checks, func(c check) bool { return c.mark == markBad }) {
		os.Exit(1)
	}
}

// doctorChecks runs every check against dir. Split from the printing so the
// verdicts — including the "broken means exit 1" contract — are testable
// without capturing stdout or exiting the test binary.
func doctorChecks(dir string) []check {
	checks := []check{checkBuild(), checkORT(), checkModel()}
	checks = append(checks, checkIndex(dir)...)
	checks = append(checks, checkSkill())
	return append(checks, checkTuning()...)
}

// checkBuild reports the binary itself: version, platform, and whether it can
// embed at all.
func checkBuild() check {
	detail := fmt.Sprintf("descry %s  %s/%s  %s", version, runtime.GOOS, runtime.GOARCH, runtime.Version())
	if !cgoEnabled {
		return check{
			mark:   markBad,
			label:  "build",
			detail: detail + "  built without cgo",
			hint:   "fix: rebuild with CGO_ENABLED=1 — embedding dlopens ONNX Runtime through cgo",
		}
	}
	return check{mark: markOK, label: "build", detail: detail + "  cgo enabled"}
}

// checkORT reports which step of the ONNX Runtime resolution chain answers,
// without downloading.
func checkORT() check {
	st := embed.LocateOnnxRuntime()
	const label = "onnxruntime"
	switch {
	case st.Err != nil:
		// The error already carries the variable or the platform, so the detail
		// is just the error; only the remedy differs between the two causes.
		hint := "fix: install onnxruntime (e.g. `brew install onnxruntime`) and set DESCRY_ORT_LIB to the shared library"
		if st.Source == embed.ORTFromEnv {
			hint = "fix: point DESCRY_ORT_LIB at the shared library, or unset it to let descry provision one"
		}
		return check{mark: markBad, label: label, detail: st.Err.Error(), hint: hint}
	case st.Source == embed.ORTNeedsDownload:
		return check{
			mark:   markInfo,
			label:  label,
			detail: fmt.Sprintf("not cached — downloads %s on first use", st.Version),
		}
	}
	return check{mark: markOK, label: label, detail: fmt.Sprintf("%s (%s)", st.Path, st.Source)}
}

// checkModel reports the selected MiniLM export and whether it is cached.
func checkModel() check {
	st, err := embed.LocateModel(os.Getenv(modelEnv))
	if err != nil {
		return check{
			mark:   markBad,
			label:  "model",
			detail: err.Error(),
			hint:   `fix: unset DESCRY_MODEL for the fp32 default, or set it to "q8"`,
		}
	}
	switch {
	case st.Cached():
		return check{mark: markOK, label: "model", detail: fmt.Sprintf("%s (cached, %s)", st.ID, humanBytes(st.ModelBytes))}
	case st.ModelCached && !st.VocabCached:
		return check{mark: markInfo, label: "model", detail: fmt.Sprintf("%s cached, vocab missing — fetched on first use", st.ID)}
	}
	// Deliberately no size: the fp32 default is ~90MB and the q8 export far
	// smaller, and a number that drifts with the upstream export is worse than
	// none.
	return check{mark: markInfo, label: "model", detail: fmt.Sprintf("%s not cached — downloaded on first index", st.ID)}
}

// checkIndex reports the repository's index and embed cache: presence, size,
// and whether the fingerprint still matches — naming the field that drifted
// when it doesn't.
func checkIndex(dir string) []check {
	dbPath := dbPathFor(dir)
	info, err := store.Inspect(dbPath)
	if os.IsNotExist(err) {
		return []check{{
			mark:   markInfo,
			label:  "index",
			detail: "no index in this repository",
			hint:   "fix: run `descry index` to build one",
		}}
	}
	if err != nil {
		return []check{{
			mark:   markBad,
			label:  "index",
			detail: fmt.Sprintf("%s: %v", dbPath, err),
			hint:   "fix: delete index.db and re-run `descry index` (keep embed_cache.db beside it)",
		}}
	}

	var out []check
	if stale := staleFields(info.Fingerprint); len(stale) > 0 {
		out = append(out, check{
			mark:   markInfo,
			label:  "index",
			detail: fmt.Sprintf("%d chunks in %s — stale: %s", info.Chunks, dbPath, strings.Join(stale, ", ")),
			hint:   "the next index or search rebuilds it automatically",
		})
	} else {
		out = append(out, check{mark: markOK, label: "index", detail: fmt.Sprintf("%d chunks in %s", info.Chunks, dbPath)})
	}

	cachePath := filepath.Join(filepath.Dir(dbPath), "embed_cache.db")
	if fi, err := os.Stat(cachePath); err == nil {
		out = append(out, check{
			mark:   markOK,
			label:  "embed cache",
			detail: fmt.Sprintf("%s — a rebuild reuses these vectors", humanBytes(fi.Size())),
		})
	} else {
		out = append(out, check{
			mark:   markInfo,
			label:  "embed cache",
			detail: "none — the next index embeds every chunk from scratch",
		})
	}
	return out
}

// staleFields names the fingerprint fields that no longer match this build, so
// the report can say *what* invalidated an index rather than only that it did.
// Two fields are deliberately not compared: dim, because it is a property of
// the embedder (already compared) and reading it would mean loading the model;
// and embedder itself when the model can't be located, since checkModel already
// reports that as its own line.
func staleFields(stored string) []string {
	if stored == "" {
		return nil // pre-fingerprint or empty index; nothing meaningful to compare
	}
	have := parseFingerprint(stored)
	want := map[string]string{
		"schema":   fmt.Sprint(store.SchemaVersion),
		"pipeline": fmt.Sprint(pipelineVersion),
	}
	// selectChunker, not a named strategy and not chunk.Default: doctor reports
	// whether the index matches what *this invocation's* pipeline would build.
	// Hardcoding a name tells every user their index is stale the day the default
	// changes; ignoring $DESCRY_CHUNKER tells the opposite lie to anyone running a
	// non-default arm — that a deliberately different index is the expected one.
	if chk, err := selectChunker(); err == nil {
		want["chunker"] = chk.ID()
	}
	if st, err := embed.LocateModel(os.Getenv(modelEnv)); err == nil {
		want["embedder"] = st.ID
	}
	var stale []string
	for k, w := range want {
		if h, ok := have[k]; ok && h != w {
			stale = append(stale, fmt.Sprintf("%s %s → %s", k, h, w))
		}
	}
	slices.Sort(stale)
	return stale
}

// parseFingerprint splits a "k=v k=v" fingerprint into its fields.
func parseFingerprint(s string) map[string]string {
	out := map[string]string{}
	for _, f := range strings.Fields(s) {
		if k, v, ok := strings.Cut(f, "="); ok {
			out[k] = v
		}
	}
	return out
}

// checkSkill reports whether coding agents have been taught to reach for descry.
func checkSkill() check {
	switch installed, stale := skill.UserState(); {
	case installed && stale:
		return check{
			mark:   markInfo,
			label:  "agent skill",
			detail: "installed but out of date",
			hint:   "fix: run `descry skill install` to refresh it",
		}
	case installed:
		return check{mark: markOK, label: "agent skill", detail: "installed and current"}
	}
	return check{
		mark:   markInfo,
		label:  "agent skill",
		detail: "not installed",
		hint:   "fix: run `descry skill install` so coding agents search with descry",
	}
}

// tuningVars are the retrieval knobs. They are swept defaults, so an override
// left in a shell silently changes every result with no other symptom — which
// is exactly the kind of thing a health report exists to surface.
var tuningVars = []string{
	"DESCRY_VEC_WEIGHT", "DESCRY_LEX_WEIGHT", "DESCRY_LEXFILE_WEIGHT",
	"DESCRY_RRF_K", "DESCRY_FUSE_ALPHA", "DESCRY_CAND_MULT",
	"DESCRY_RERANK_MULT", "DESCRY_BM25_PATH",
}

// checkTuning reports non-default retrieval settings, and stays silent when
// there are none.
func checkTuning() []check {
	var set []string
	for _, v := range tuningVars {
		if val := os.Getenv(v); val != "" {
			set = append(set, fmt.Sprintf("%s=%s", v, val))
		}
	}
	if len(set) == 0 {
		return nil
	}
	return []check{{
		mark:   markInfo,
		label:  "tuning",
		detail: strings.Join(set, " "),
		hint:   "retrieval defaults are overridden — unset these to restore the swept optimum",
	}}
}

// humanBytes formats a file size for a one-line report.
func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for n/div >= unit && exp < 3 {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGT"[exp])
}
