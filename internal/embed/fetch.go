// Download-on-first-use for the MiniLM model files: one network call ever, then
// a local cache under the user cache dir.
package embed

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// minilmVocabURL is the WordPiece vocabulary, shared by every MiniLM export.
const minilmVocabURL = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/vocab.txt"

// modelSpec pins one MiniLM export: where it downloads from, the file it is
// cached as, and the embedder identity its vector space belongs to.
type modelSpec struct {
	url    string
	sha256 string // "" = unpinned (the fp32 default tracks the repo's main revision)
	file   string
	id     string
}

// modelSpecs maps DESCRY_MODEL values to exports. "" is the fp32 default (ONNX
// Runtime imposes no opset ceiling, so this is a free choice; the older pure-Go
// runtime restricted it). "q8" is the int8-quantized export — smaller and ~1.9x
// faster to index at a small quality cost — pinned to an immutable revision and
// sha256-verified.
var modelSpecs = map[string]modelSpec{
	"": {
		url:  "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
		file: "all-MiniLM-L6-v2.onnx",
		id:   "all-MiniLM-L6-v2",
	},
	"q8": {
		url:    "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/onnx/model_quantized.onnx",
		sha256: "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1",
		file:   "all-MiniLM-L6-v2-q8.onnx",
		id:     "all-MiniLM-L6-v2-q8",
	},
}

// EnsureModel returns local paths to the MiniLM export selected by name (a
// DESCRY_MODEL value: "" for the fp32 default, "q8" for the quantized export)
// and its WordPiece vocab, downloading either to the user cache dir if missing.
// The ~90MB model download happens once, on the first indexing run; pinned
// exports are sha256-verified, and a mismatched file is deleted and rejected.
// The returned id is the embedder identity for the model's vector space (see
// NewOrtEmbedderID).
func EnsureModel(name string) (modelPath, vocabPath, id string, err error) {
	spec, ok := modelSpecs[name]
	if !ok {
		return "", "", "", fmt.Errorf("unknown DESCRY_MODEL %q (valid: unset for fp32, \"q8\")", name)
	}
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", "", "", err
	}
	dir := filepath.Join(cache, "descry", "models")
	modelPath = filepath.Join(dir, spec.file)
	vocabPath = filepath.Join(dir, "vocab.txt")
	if err := downloadOnce(spec.url, modelPath); err != nil {
		return "", "", "", fmt.Errorf("fetch model: %w", err)
	}
	if spec.sha256 != "" {
		if err := verifySHA256(modelPath, spec.sha256); err != nil {
			os.Remove(modelPath)
			return "", "", "", fmt.Errorf("model integrity: %w", err)
		}
	}
	if err := downloadOnce(minilmVocabURL, vocabPath); err != nil {
		return "", "", "", fmt.Errorf("fetch vocab: %w", err)
	}
	return modelPath, vocabPath, spec.id, nil
}

// verifySHA256 checks a file against an expected hex digest.
func verifySHA256(path, want string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("sha256 mismatch: got %s, want %s", got, want)
	}
	return nil
}

// downloadOnce fetches url to dest if it isn't already cached, writing through
// a temp file so a partial download never looks complete.
func downloadOnce(url, dest string) error {
	if _, err := os.Stat(dest); err == nil {
		return nil // already cached
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: unexpected status %s", url, resp.Status)
	}
	tmp := dest + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dest)
}
