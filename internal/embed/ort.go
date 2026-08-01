// OrtEmbedder runs the all-MiniLM-L6-v2 embedding pipeline on ONNX Runtime
// (Microsoft's C++ engine) via cgo: WordPiece tokenize (wordpiece.go) → forward
// pass → mask-weighted mean pooling + L2 normalization (pool.go), the standard
// sentence-transformers recipe.
//
// The onnxruntime shared library is loaded at runtime (dlopen), not linked, so
// it is provisioned on first use by EnsureOnnxRuntime (ortlib.go) — no manual
// install needed. Building does require cgo (a C compiler).
package embed

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	// minilmDim is the width of the default MiniLM export, and the width
	// NewOrtEmbedder assumes. Other exports declare their own — see modelSpec.
	minilmDim     = 384
	onnxMaxTokens = 256 // plenty for declaration-sized AST chunks
)

// ortInitOnce guards ONNX Runtime's process-global environment. The library
// can only be initialized once, and every session shares it.
var (
	ortInitOnce sync.Once
	ortInitErr  error
)

// initORT resolves the shared library (see EnsureOnnxRuntime) and initializes
// the process-global runtime environment.
func initORT() error {
	ortInitOnce.Do(func() {
		lib, err := EnsureOnnxRuntime()
		if err != nil {
			ortInitErr = err
			return
		}
		ort.SetSharedLibraryPath(lib)
		ortInitErr = ort.InitializeEnvironment()
	})
	if ortInitErr != nil {
		return fmt.Errorf("initialize onnxruntime: %w", ortInitErr)
	}
	return nil
}

// OrtEmbedder implements Embedder over an ONNX Runtime session. Safe for
// concurrent use: ORT sessions support concurrent Run calls by design (each
// call carries its own input/output tensors).
type OrtEmbedder struct {
	session *ort.DynamicAdvancedSession
	tok     *WordPiece
	id      string
	dim     int
	pool    Pooling
}

// Pooling names how a model's token vectors collapse into one sentence vector.
// It belongs to the model, not to descry: see clsPool.
type Pooling string

const (
	PoolMean Pooling = "mean" // sentence-transformers recipe (MiniLM)
	PoolCLS  Pooling = "cls"  // BERT [CLS] token (the BGE family)
)

// NewOrtEmbedder loads the fp32 MiniLM graph and runs a warm-up inference, so a
// missing or broken native library fails loudly here rather than mid-index.
func NewOrtEmbedder(modelPath, vocabPath string) (*OrtEmbedder, error) {
	return NewOrtEmbedderID(modelPath, vocabPath, "all-MiniLM-L6-v2", minilmDim, PoolMean)
}

// NewOrtEmbedderID loads an ONNX model under an explicit embedder identity and
// output width. Use a distinct id for any file that is not vector-identical to
// the fp32 default (e.g. the q8 quantized export) — the id feeds the index
// fingerprint and the embed-cache key, and mixing vector spaces under one id
// corrupts both.
//
// dim must match the model's hidden size and pool must be the recipe the model
// was trained with. Both are parameters rather than constants because comparing
// a candidate model against the shipped one is a routine question ("is a bigger
// encoder worth the index time?"), and hardcoding either turns that experiment
// into a refactor — or, worse, into a measurement of the wrong thing.
func NewOrtEmbedderID(modelPath, vocabPath, id string, dim int, pool Pooling) (*OrtEmbedder, error) {
	tok, err := LoadWordPiece(vocabPath, onnxMaxTokens)
	if err != nil {
		return nil, fmt.Errorf("load vocab: %w", err)
	}
	if err := initORT(); err != nil {
		return nil, err
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("ort session options: %w", err)
	}
	defer opts.Destroy()
	// One thread per inference: the index worker pool already runs one embed
	// per core, so intra-op threads would only fight it for CPUs. Per-chunk
	// parallelism beats per-matmul parallelism at our short sequence lengths.
	if err := opts.SetIntraOpNumThreads(1); err != nil {
		return nil, fmt.Errorf("ort thread config: %w", err)
	}
	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"}, opts)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	e := &OrtEmbedder{session: session, tok: tok, id: id, dim: dim, pool: pool}
	if _, err := e.embed("warm up"); err != nil {
		session.Destroy()
		return nil, fmt.Errorf("model warm-up: %w", err)
	}
	return e, nil
}

// Dim reports the model's embedding width.
func (e *OrtEmbedder) Dim() int { return e.dim }

// ID reports the embedder identity set at construction, which feeds the index
// fingerprint and the embed-cache key. The fp32 default keeps the historical id
// "all-MiniLM-L6-v2" — the weights and recipe are unchanged from the previous
// pure-Go runtime (vectors matched to cosine > 0.9999), so existing indexes and
// caches stay valid and dropping that runtime does NOT force a reindex.
func (e *OrtEmbedder) ID() string { return e.id }

// Embed produces the L2-normalized sentence embedding for text.
func (e *OrtEmbedder) Embed(text string) []float32 {
	v, err := e.embed(text)
	if err != nil {
		// Warm-up proved the session runs; a nil vector scores 0 everywhere.
		return nil
	}
	return v
}

func (e *OrtEmbedder) embed(text string) ([]float32, error) {
	enc := e.tok.Encode(text)
	n := int64(len(enc.IDs))
	shape := ort.NewShape(1, n)

	ids, err := ort.NewTensor(shape, enc.IDs)
	if err != nil {
		return nil, fmt.Errorf("input_ids tensor: %w", err)
	}
	defer ids.Destroy()
	mask, err := ort.NewTensor(shape, enc.Mask)
	if err != nil {
		return nil, fmt.Errorf("attention_mask tensor: %w", err)
	}
	defer mask.Destroy()
	types, err := ort.NewTensor(shape, enc.TypeIDs)
	if err != nil {
		return nil, fmt.Errorf("token_type_ids tensor: %w", err)
	}
	defer types.Destroy()

	outputs := []ort.Value{nil} // let ORT allocate last_hidden_state
	if err := e.session.Run([]ort.Value{ids, mask, types}, outputs); err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}
	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		outputs[0].Destroy()
		return nil, fmt.Errorf("unexpected output type %T", outputs[0])
	}
	defer out.Destroy()
	data := out.GetData()
	if len(data) != int(n)*e.dim {
		return nil, fmt.Errorf("unexpected output length %d, want %d", len(data), int(n)*e.dim)
	}
	// Both poolers copy into a fresh slice, so freeing the tensor after is safe.
	if e.pool == PoolCLS {
		return l2norm(clsPool(data, e.dim)), nil
	}
	return l2norm(meanPool(data, enc.Mask, e.dim)), nil
}

// Close releases the ORT session.
func (e *OrtEmbedder) Close() error { return e.session.Destroy() }

// NewSemanticEmbedder returns the default semantic embedder: fp32 MiniLM on
// ONNX Runtime.
func NewSemanticEmbedder(modelPath, vocabPath string) (Embedder, error) {
	return NewOrtEmbedder(modelPath, vocabPath)
}

// NewSemanticEmbedderQ8 returns the int8-quantized MiniLM export — roughly
// half the weight bytes and ~1.9x faster to index, at a sub-1pt quality cost
// (README "Measured quality"). It runs under its own identity so its vector
// space never mixes with fp32's.
func NewSemanticEmbedderQ8(modelPath, vocabPath string) (Embedder, error) {
	return NewOrtEmbedderID(modelPath, vocabPath, "all-MiniLM-L6-v2-q8", minilmDim, PoolMean)
}

var _ Embedder = (*OrtEmbedder)(nil)
