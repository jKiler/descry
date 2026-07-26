// Provisioning for the ONNX Runtime shared library.
//
// onnxruntime_go dlopens the library at runtime rather than linking it, so the
// library does not need to be present to build — only to run. That lets descry
// fetch it on first use the same way it fetches the model, instead of making
// every user install it through a package manager.
package embed

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// ortVersion is the pinned ONNX Runtime release. Bump it together with every
// digest in ortAssets.
const ortVersion = "1.27.1"

// ortAsset is one platform's release archive: the file name under the release
// tag, and its sha256 as published by the release itself.
type ortAsset struct {
	name   string
	sha256 string
}

// ortAssets maps GOOS/GOARCH to the official release archive. Digests come from
// the microsoft/onnxruntime v1.27.1 release assets; a download that doesn't
// match is rejected, so a corrupted or substituted archive is never loaded.
//
// darwin/amd64 is absent because the 1.27.1 release publishes no macOS x86_64
// build — those users install the library themselves and point DESCRY_ORT_LIB
// at it.
var ortAssets = map[string]ortAsset{
	"darwin/arm64":  {"onnxruntime-osx-arm64-" + ortVersion + ".tgz", "e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6"},
	"linux/amd64":   {"onnxruntime-linux-x64-" + ortVersion + ".tgz", "25b1ef1fea1acd210d63f8f24dc870ad6e077795ce1f54876252c6d3803c15af"},
	"linux/arm64":   {"onnxruntime-linux-aarch64-" + ortVersion + ".tgz", "33c67e33d1e25b816878366ea276589a024f71f000e7ff955c4b33224d639edd"},
	"windows/amd64": {"onnxruntime-win-x64-" + ortVersion + ".zip", "2e00414a63fdef0914cd5a5ede6c707844878e0c08e1b6693842f0451b2df2a1"},
	"windows/arm64": {"onnxruntime-win-arm64-" + ortVersion + ".zip", "6e22c2061ba6400b42a59663d700c8694e4e8fe654cf452c4700c24237407ae1"},
}

// systemLibPaths are conventional install locations, checked before downloading
// so a machine that already has the library doesn't pay for a copy.
var systemLibPaths = []string{
	"/opt/homebrew/lib/libonnxruntime.dylib", // macOS arm64 (Homebrew)
	"/usr/local/lib/libonnxruntime.dylib",    // macOS x86_64 (Homebrew)
	"/usr/local/lib/libonnxruntime.so",
	"/usr/lib/libonnxruntime.so",
	"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
}

// ORTSource names which step of the resolution chain answered.
type ORTSource string

const (
	ORTFromEnv       ORTSource = "DESCRY_ORT_LIB"  // explicit override
	ORTFromCache     ORTSource = "cached download" // downloaded here earlier
	ORTFromSystem    ORTSource = "system install"  // Homebrew, /usr/lib, …
	ORTNeedsDownload ORTSource = "not cached"      // would download on first use

	// ORTUnsupported means descry cannot provision the library here: either no
	// pinned build exists for this platform, or there is nowhere to cache one.
	// Either way the user must install onnxruntime and set DESCRY_ORT_LIB.
	ORTUnsupported ORTSource = "unsupported"
)

// ORTState is where the onnxruntime library would come from, resolved without
// touching the network. Err is set only for a *broken* state (a DESCRY_ORT_LIB
// pointing at nothing, an unreadable cache dir, an unsupported platform);
// ORTNeedsDownload is a healthy state that simply hasn't happened yet.
type ORTState struct {
	Source  ORTSource
	Path    string // resolved library, or where a download would land
	URL     string // release archive, when Source is ORTNeedsDownload
	Version string
	Err     error

	// asset is the pinned release entry behind URL. Carried rather than
	// re-derived by the downloader, so the archive name and — the part that
	// matters — the digest it is verified against can never be looked up from a
	// different map entry than the URL was.
	asset ortAsset
}

// LocateOnnxRuntime reports where the onnxruntime shared library resolves from,
// without downloading anything. It is the read-only half of EnsureOnnxRuntime,
// which is defined in terms of it so the two can't describe different chains.
func LocateOnnxRuntime() ORTState {
	st := ORTState{Version: ortVersion}

	if p := os.Getenv("DESCRY_ORT_LIB"); p != "" {
		st.Source, st.Path = ORTFromEnv, p
		if _, err := os.Stat(p); err != nil {
			// The stat error already names the path; don't repeat it.
			st.Err = fmt.Errorf("DESCRY_ORT_LIB: %w", err)
		}
		return st
	}

	// The cache dir is only needed for steps 2 and 4, so a machine where it
	// can't be resolved (no HOME, as in a minimal container) still gets the
	// benefit of a system install.
	dir, cacheErr := ortCacheDir()
	var cached string
	if cacheErr == nil {
		cached = filepath.Join(dir, ortLibName())
		if _, err := os.Stat(cached); err == nil {
			st.Source, st.Path = ORTFromCache, cached
			return st
		}
	}

	for _, p := range systemLibPaths {
		if _, err := os.Stat(p); err == nil {
			st.Source, st.Path = ORTFromSystem, p
			return st
		}
	}

	asset, ok := ortAssets[runtime.GOOS+"/"+runtime.GOARCH]
	if !ok {
		st.Source = ORTUnsupported
		st.Err = fmt.Errorf("no pinned ONNX Runtime build for %s/%s: install onnxruntime and set DESCRY_ORT_LIB to the shared library",
			runtime.GOOS, runtime.GOARCH)
		return st
	}
	if cacheErr != nil {
		// A pinned build exists, but there is nowhere to cache it.
		st.Source, st.Err = ORTUnsupported, cacheErr
		return st
	}
	st.Source = ORTNeedsDownload
	st.Path = cached
	st.asset = asset
	st.URL = "https://github.com/microsoft/onnxruntime/releases/download/v" + ortVersion + "/" + asset.name
	return st
}

// EnsureOnnxRuntime returns a path to the onnxruntime shared library, resolving
// in this order:
//
//  1. DESCRY_ORT_LIB — an explicit override always wins;
//  2. a copy this function downloaded earlier (cached under the user cache dir);
//  3. a system install (Homebrew, /usr/lib, …);
//  4. the pinned release for this platform, downloaded and sha256-verified.
//
// Only step 4 touches the network, and only once per machine. Use
// LocateOnnxRuntime to inspect the same chain without downloading.
func EnsureOnnxRuntime() (string, error) {
	st := LocateOnnxRuntime()
	if st.Err != nil {
		return "", st.Err
	}
	if st.Source != ORTNeedsDownload {
		return st.Path, nil
	}

	cached := st.Path
	dir := filepath.Dir(cached)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	archive := filepath.Join(dir, st.asset.name)
	fmt.Fprintf(os.Stderr, "descry: downloading ONNX Runtime %s for %s/%s (one time)\n", ortVersion, runtime.GOOS, runtime.GOARCH)
	if err := downloadOnce(st.URL, archive); err != nil {
		return "", fmt.Errorf("download onnxruntime: %w", err)
	}
	if err := verifySHA256(archive, st.asset.sha256); err != nil {
		os.Remove(archive)
		return "", fmt.Errorf("onnxruntime integrity: %w", err)
	}
	if err := extractOrtLib(archive, cached); err != nil {
		return "", fmt.Errorf("extract onnxruntime: %w", err)
	}
	os.Remove(archive) // the library is extracted; the archive is just cache weight
	return cached, nil
}

// ortCacheDir is where the downloaded library lives, beside the model cache.
func ortCacheDir() (string, error) {
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("user cache dir: %w", err)
	}
	return filepath.Join(cache, "descry", "runtime", ortVersion), nil
}

// ortLibName is the platform's shared-library file name.
func ortLibName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime.dylib"
	case "windows":
		return "onnxruntime.dll"
	default:
		return "libonnxruntime.so"
	}
}

// extractOrtLib pulls the single shared library out of the release archive and
// writes it to dst. Release archives nest it under <archive-root>/lib/.
func extractOrtLib(archive, dst string) error {
	if strings.HasSuffix(archive, ".zip") {
		return extractFromZip(archive, dst)
	}
	return extractFromTarGz(archive, dst)
}

// isOrtLib reports whether a path inside an archive is the shared library we
// want — the real object, not a version symlink or an import library.
func isOrtLib(name string) bool {
	base := filepath.Base(name)
	if !strings.Contains(filepath.ToSlash(name), "/lib/") {
		return false
	}
	switch runtime.GOOS {
	case "darwin":
		// libonnxruntime.1.27.1.dylib (the versioned real file)
		return strings.HasPrefix(base, "libonnxruntime") && strings.HasSuffix(base, ".dylib")
	case "windows":
		return base == "onnxruntime.dll"
	default:
		// libonnxruntime.so.1.27.1
		return strings.HasPrefix(base, "libonnxruntime.so")
	}
}

func extractFromTarGz(archive, dst string) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return fmt.Errorf("no shared library found in %s", filepath.Base(archive))
		}
		if err != nil {
			return err
		}
		// Only regular files: a symlink entry would otherwise "win" over the
		// real object and produce a dangling library.
		if h.Typeflag != tar.TypeReg || !isOrtLib(h.Name) {
			continue
		}
		return writeLib(dst, tr)
	}
}

func extractFromZip(archive, dst string) error {
	zr, err := zip.OpenReader(archive)
	if err != nil {
		return err
	}
	defer zr.Close()
	for _, f := range zr.File {
		if f.FileInfo().IsDir() || !isOrtLib(f.Name) {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		defer rc.Close()
		return writeLib(dst, rc)
	}
	return fmt.Errorf("no shared library found in %s", filepath.Base(archive))
}

// writeLib streams the library to dst through a temp file, so an interrupted
// extraction never leaves a half-written library that later looks cached.
func writeLib(dst string, r io.Reader) error {
	tmp := dst + ".tmp"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, r); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}
