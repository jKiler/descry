package embed

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// ortTestPaths returns the cached model + the vendored test vocab, skipping when
// the model isn't downloaded so a fresh checkout stays hermetic.
func ortTestPaths(t *testing.T) (string, string) {
	t.Helper()
	cache, err := os.UserCacheDir()
	if err != nil {
		t.Skip("no user cache dir")
	}
	mp := filepath.Join(cache, "descry", "models", "all-MiniLM-L6-v2.onnx")
	if _, err := os.Stat(mp); err != nil {
		t.Skip("model not cached; run `descry index` once to download it")
	}
	return mp, "testdata/vocab.txt"
}

func cosine32(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// The contract of the semantic embedder: fixed width, deterministic, and — the
// reason it exists — paraphrases score higher than unrelated text on MEANING,
// which a purely lexical embedder can't do.
func TestSemanticEmbedder(t *testing.T) {
	mp, vp := ortTestPaths(t)
	e, err := NewSemanticEmbedder(mp, vp)
	if err != nil {
		t.Fatalf("NewSemanticEmbedder: %v", err)
	}

	t.Run("dimension", func(t *testing.T) {
		if e.Dim() != onnxDim {
			t.Errorf("Dim() = %d, want %d", e.Dim(), onnxDim)
		}
		if got := len(e.Embed("hello world")); got != onnxDim {
			t.Errorf("embedding width = %d, want %d", got, onnxDim)
		}
	})

	t.Run("deterministic", func(t *testing.T) {
		a, b := e.Embed("the quick brown fox"), e.Embed("the quick brown fox")
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("same text embedded differently at %d: %v vs %v", i, a[i], b[i])
			}
		}
	})

	t.Run("semantic similarity beats lexical overlap", func(t *testing.T) {
		// These pairs share no content words ("db"/"database"), so only a real
		// model can rank the paraphrase above the unrelated sentence.
		query := e.Embed("database connection error")
		paraphrase := e.Embed("failed to connect to the db")
		unrelated := e.Embed("the weather is sunny today")

		simPara, simUnrel := cosine32(query, paraphrase), cosine32(query, unrelated)
		if simPara <= simUnrel {
			t.Errorf("paraphrase %.4f should score above unrelated %.4f", simPara, simUnrel)
		}
	})
}

// TestOrtQ8DistinctSpace pins the q8 embedder's contract: a distinct ID (so it
// never shares a fingerprint or embed cache with fp32) over normalized 384-dim
// vectors that are close to — but deliberately not identical to — fp32's, since
// int8 quantization shifts the vector space slightly.
func TestOrtQ8DistinctSpace(t *testing.T) {
	_, vp := ortTestPaths(t)
	cache, _ := os.UserCacheDir()
	q8Path := filepath.Join(cache, "descry", "models", "all-MiniLM-L6-v2-q8.onnx")
	if _, err := os.Stat(q8Path); err != nil {
		t.Skip("q8 model not cached; run `DESCRY_MODEL=q8 descry index` once")
	}
	q8, err := NewSemanticEmbedderQ8(q8Path, vp)
	if err != nil {
		t.Fatal(err)
	}
	if q8.ID() == "all-MiniLM-L6-v2" {
		t.Error("q8 must not share the fp32 embedder ID — it would corrupt the shared cache/index")
	}

	mp, _ := ortTestPaths(t)
	fp32, err := NewOrtEmbedder(mp, vp)
	if err != nil {
		t.Fatal(err)
	}
	defer fp32.Close()

	text := "rank documents by keyword relevance"
	qv, fv := q8.Embed(text), fp32.Embed(text)
	if len(qv) != onnxDim {
		t.Fatalf("q8 dim = %d, want %d", len(qv), onnxDim)
	}
	var norm float64
	for _, x := range qv {
		norm += float64(x) * float64(x)
	}
	if math.Abs(norm-1.0) > 1e-3 {
		t.Errorf("q8 vector not L2-normalized: |v|² = %.4f", norm)
	}
	// Same model, quantized: close but not bit-identical.
	if cos := cosine32(qv, fv); cos < 0.95 || cos >= 0.99999 {
		t.Errorf("q8 vs fp32 cosine %.5f: want close (>0.95) but distinct (<0.99999)", cos)
	}
}

// Informational: per-embed latency of the runtime.
func TestOrtLatency(t *testing.T) {
	mp, vp := ortTestPaths(t)
	e, err := NewOrtEmbedder(mp, vp)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	text := "persist chunks and vectors in a sqlite database file so the index survives restarts"
	e.Embed(text) // warm
	const reps = 5
	t0 := time.Now()
	for range reps {
		e.Embed(text)
	}
	t.Logf("onnxruntime: %v/embed", time.Since(t0)/reps)
}

// An explicit DESCRY_ORT_LIB always wins, and a bad one is reported rather than
// silently ignored (which would send resolution down the download path).
func TestEnsureOnnxRuntimeOverride(t *testing.T) {
	lib := filepath.Join(t.TempDir(), "libonnxruntime.dylib")
	if err := os.WriteFile(lib, []byte("not a real library"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("DESCRY_ORT_LIB", lib)
	got, err := EnsureOnnxRuntime()
	if err != nil || got != lib {
		t.Errorf("EnsureOnnxRuntime() = (%q, %v), want (%q, nil)", got, err, lib)
	}

	t.Setenv("DESCRY_ORT_LIB", filepath.Join(t.TempDir(), "missing.dylib"))
	if _, err := EnsureOnnxRuntime(); err == nil {
		t.Error("a DESCRY_ORT_LIB pointing at a missing file should error")
	}
}

// The archive-entry matcher must never pick headers, docs, or files outside
// lib/ — extracting the wrong entry would produce an unloadable library.
func TestIsOrtLibRejectsNonLibraries(t *testing.T) {
	if ortLibName() == "" {
		t.Fatal("ortLibName() is empty")
	}
	root := "onnxruntime-osx-arm64-1.27.1/"
	for _, name := range []string{
		root + "include/onnxruntime_c_api.h",
		root + "README.md",
		root + "GIT_COMMIT_ID",
		root + "libonnxruntime.dylib", // not under lib/
		root + "lib/cmake/config.cmake",
	} {
		if isOrtLib(name) {
			t.Errorf("isOrtLib(%q) = true, want false", name)
		}
	}
	// And it must accept this platform's real library entry.
	var real string
	switch {
	case ortLibName() == "libonnxruntime.dylib":
		real = root + "lib/libonnxruntime.1.27.1.dylib"
	case ortLibName() == "onnxruntime.dll":
		real = root + "lib/onnxruntime.dll"
	default:
		real = root + "lib/libonnxruntime.so.1.27.1"
	}
	if !isOrtLib(real) {
		t.Errorf("isOrtLib(%q) = false, want true", real)
	}
}
