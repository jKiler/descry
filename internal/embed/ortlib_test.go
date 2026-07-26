package embed

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// makeTarGz builds an archive shaped like a real onnxruntime release: the real
// versioned library under lib/, plus a symlink and non-library files that must
// be ignored.
func makeTarGz(t *testing.T, path, libEntry string, libBody []byte) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz := gzip.NewWriter(f)
	defer gz.Close()
	tw := tar.NewWriter(gz)
	defer tw.Close()

	root := "onnxruntime-osx-arm64-1.27.1/"
	// A symlink entry that points at the real file — must NOT be extracted,
	// or we'd write a dangling stub and "successfully" cache garbage.
	if err := tw.WriteHeader(&tar.Header{
		Name: root + "lib/" + ortLibName(), Typeflag: tar.TypeSymlink,
		Linkname: filepath.Base(libEntry), Mode: 0o777,
	}); err != nil {
		t.Fatal(err)
	}
	for _, junk := range []string{root + "README.md", root + "include/onnxruntime_c_api.h"} {
		if err := tw.WriteHeader(&tar.Header{Name: junk, Typeflag: tar.TypeReg, Size: 3, Mode: 0o644}); err != nil {
			t.Fatal(err)
		}
		tw.Write([]byte("xxx"))
	}
	if err := tw.WriteHeader(&tar.Header{
		Name: root + "lib/" + libEntry, Typeflag: tar.TypeReg, Size: int64(len(libBody)), Mode: 0o755,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write(libBody); err != nil {
		t.Fatal(err)
	}
}

// Extraction must pull the real shared object out of a release-shaped tarball,
// skipping the symlink and the non-library files.
func TestExtractOrtLibFromTarGz(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "onnxruntime.tgz")

	var entry string
	switch runtime.GOOS {
	case "darwin":
		entry = "libonnxruntime.1.27.1.dylib"
	case "windows":
		entry = "onnxruntime.dll"
	default:
		entry = "libonnxruntime.so.1.27.1"
	}
	body := []byte("ELF-ish shared library payload")
	makeTarGz(t, archive, entry, body)

	dst := filepath.Join(dir, ortLibName())
	if err := extractOrtLib(archive, dst); err != nil {
		t.Fatalf("extractOrtLib: %v", err)
	}
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("extracted library missing: %v", err)
	}
	if string(got) != string(body) {
		t.Errorf("extracted %q, want the real library body %q", got, body)
	}
	fi, err := os.Stat(dst)
	if err != nil || fi.Mode().Perm()&0o100 == 0 {
		t.Errorf("extracted library should be executable, mode = %v", fi.Mode())
	}
}

// An archive with no library must fail loudly rather than leave a partial file.
func TestExtractOrtLibMissingLibrary(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "empty.tgz")
	f, _ := os.Create(archive)
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	tw.WriteHeader(&tar.Header{Name: "onnxruntime/README.md", Typeflag: tar.TypeReg, Size: 1, Mode: 0o644})
	tw.Write([]byte("x"))
	tw.Close()
	gz.Close()
	f.Close()

	dst := filepath.Join(dir, ortLibName())
	if err := extractOrtLib(archive, dst); err == nil {
		t.Error("expected an error when the archive contains no library")
	}
	if _, err := os.Stat(dst); err == nil {
		t.Error("no library file should be left behind on failure")
	}
}

// The Windows path uses zip; exercise that branch too.
func TestExtractOrtLibFromZip(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "onnxruntime.zip")
	f, err := os.Create(archive)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	for _, n := range []string{"onnxruntime-win-x64-1.27.1/README.md", "onnxruntime-win-x64-1.27.1/lib/onnxruntime.dll"} {
		w, err := zw.Create(n)
		if err != nil {
			t.Fatal(err)
		}
		w.Write([]byte("payload-" + filepath.Base(n)))
	}
	zw.Close()
	f.Close()

	if runtime.GOOS != "windows" {
		// isOrtLib is platform-specific; on non-Windows the dll isn't the target,
		// so extraction correctly reports "not found".
		if err := extractOrtLib(archive, filepath.Join(dir, ortLibName())); err == nil {
			t.Error("a windows archive should not satisfy a non-windows platform")
		}
		return
	}
	dst := filepath.Join(dir, ortLibName())
	if err := extractOrtLib(archive, dst); err != nil {
		t.Fatalf("extractOrtLib(zip): %v", err)
	}
}

// Every pinned asset must have a plausible sha256, so a typo can't disable
// verification silently.
func TestOrtAssetsHaveDigests(t *testing.T) {
	if len(ortAssets) == 0 {
		t.Fatal("no pinned ONNX Runtime assets")
	}
	for plat, a := range ortAssets {
		if len(a.sha256) != 64 {
			t.Errorf("%s: sha256 %q is not 64 hex chars", plat, a.sha256)
		}
		if a.name == "" {
			t.Errorf("%s: empty asset name", plat)
		}
	}
}

// Opt-in integration check (DESCRY_TEST_DOWNLOAD=1): fetch the real pinned
// release for this platform and verify the URL, the published sha256, and the
// archive layout assumption all hold. Skipped by default — it downloads tens of
// megabytes — but this is the only thing that proves the "no manual install"
// promise for a machine without onnxruntime.
func TestDownloadRealOrtRelease(t *testing.T) {
	if os.Getenv("DESCRY_TEST_DOWNLOAD") != "1" {
		t.Skip("set DESCRY_TEST_DOWNLOAD=1 to exercise the real download")
	}
	asset, ok := ortAssets[runtime.GOOS+"/"+runtime.GOARCH]
	if !ok {
		t.Skipf("no pinned asset for %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	dir := t.TempDir()
	archive := filepath.Join(dir, asset.name)
	url := "https://github.com/microsoft/onnxruntime/releases/download/v" + ortVersion + "/" + asset.name

	if err := downloadOnce(url, archive); err != nil {
		t.Fatalf("download %s: %v", url, err)
	}
	if err := verifySHA256(archive, asset.sha256); err != nil {
		t.Fatalf("pinned digest does not match the published asset: %v", err)
	}
	dst := filepath.Join(dir, ortLibName())
	if err := extractOrtLib(archive, dst); err != nil {
		t.Fatalf("extract from the real archive: %v", err)
	}
	fi, err := os.Stat(dst)
	if err != nil {
		t.Fatalf("no library extracted: %v", err)
	}
	if fi.Size() < 1<<20 {
		t.Errorf("extracted library is only %d bytes — probably a symlink, not the real object", fi.Size())
	}
	t.Logf("extracted %s (%.1f MB) from the real release", ortLibName(), float64(fi.Size())/1e6)
}
