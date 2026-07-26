package embed

import (
	"os"
	"path/filepath"
	"testing"
)

// TestLocateOnnxRuntimeHonorsOverride pins step 1 of the resolution chain: an
// explicit DESCRY_ORT_LIB wins outright, without consulting cache or system.
func TestLocateOnnxRuntimeHonorsOverride(t *testing.T) {
	lib := filepath.Join(t.TempDir(), "libonnxruntime.dylib")
	if err := os.WriteFile(lib, []byte("not really a library"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("DESCRY_ORT_LIB", lib)

	st := LocateOnnxRuntime()
	if st.Err != nil {
		t.Fatalf("unexpected error: %v", st.Err)
	}
	if st.Source != ORTFromEnv || st.Path != lib {
		t.Errorf("got source %q path %q, want %q %q", st.Source, st.Path, ORTFromEnv, lib)
	}
}

// TestLocateOnnxRuntimeReportsBrokenOverride: a DESCRY_ORT_LIB pointing at
// nothing is the misconfiguration doctor exists to catch, and it must surface
// as an error rather than silently falling through to a 100MB download.
func TestLocateOnnxRuntimeReportsBrokenOverride(t *testing.T) {
	t.Setenv("DESCRY_ORT_LIB", filepath.Join(t.TempDir(), "absent.dylib"))

	st := LocateOnnxRuntime()
	if st.Err == nil {
		t.Fatal("a missing DESCRY_ORT_LIB target should report an error")
	}
	if st.Source != ORTFromEnv {
		t.Errorf("source = %q, want %q so the report can name the variable", st.Source, ORTFromEnv)
	}
}

// TestLocateModelDoesNotDownload pins the read-only contract: probing an
// uncached model reports paths and stays off the network. The cache dir is
// redirected so a developer's real ~90MB cache can't make this pass by accident.
func TestLocateModelDoesNotDownload(t *testing.T) {
	// Every variable UserCacheDir consults, so the redirect holds on all three
	// platforms: XDG_CACHE_HOME on linux, $HOME/Library/Caches on darwin,
	// %LocalAppData% on windows. Missing one would let this test run against
	// the developer's real cache — passing vacuously, or failing because the
	// model happens to be there.
	cache := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cache)
	t.Setenv("HOME", cache)
	t.Setenv("LOCALAPPDATA", cache)

	st, err := LocateModel("")
	if err != nil {
		t.Fatalf("locate: %v", err)
	}
	if st.Cached() {
		t.Error("a fresh cache dir should not report the model as cached")
	}
	if st.ID == "" || st.ModelPath == "" || st.VocabPath == "" {
		t.Errorf("locate should report identity and paths even when uncached: %+v", st)
	}
	// downloadOnce only ever writes to its destination, so "no file at
	// ModelPath" is a sound proxy for "made no request".
	if _, err := os.Stat(st.ModelPath); err == nil {
		t.Error("LocateModel downloaded the model; it must only report")
	}
}

func TestLocateModelRejectsUnknownName(t *testing.T) {
	if _, err := LocateModel("nope"); err == nil {
		t.Error("an unknown DESCRY_MODEL should be an error naming the valid values")
	}
}
