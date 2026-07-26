package index

import (
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/jKiler/descry/internal/core"
	"github.com/jKiler/descry/internal/embed"
)

// embedChunks parallelizes embedding across CPU cores; the result must be
// byte-identical to a plain serial loop — same vectors, same order. Uses the
// fast HashEmbedder so the test needs no model download. (Concurrent safety of
// the ONNX embedder itself is proven in internal/embed under -race.)
func TestEmbedChunks_MatchesSerial(t *testing.T) {
	emb := embed.NewHashEmbedder(64)

	chunks := make([]core.Chunk, 200)
	for i := range chunks {
		chunks[i] = core.Chunk{Content: fmt.Sprintf("chunk %d with a few searchable words here", i)}
	}

	got := append([]core.Chunk(nil), chunks...)
	embedChunks(emb, got, nil)

	for i := range got {
		want := emb.Embed(chunks[i].Content)
		if !reflect.DeepEqual(got[i].Vector, want) {
			t.Fatalf("chunk %d: parallel vector != serial vector", i)
		}
	}
}

// The progress callback's contract: an initial (0, total), one call per chunk,
// and a final count equal to total — regardless of worker interleaving.
func TestEmbedChunks_Progress(t *testing.T) {
	emb := embed.NewHashEmbedder(8)
	chunks := make([]core.Chunk, 50)
	for i := range chunks {
		chunks[i] = core.Chunk{Content: fmt.Sprintf("chunk %d", i)}
	}

	var mu sync.Mutex
	var calls []int
	sawTotal := 50
	embedChunks(emb, chunks, func(done, total int) {
		mu.Lock()
		defer mu.Unlock()
		if total != sawTotal {
			t.Errorf("total = %d, want %d", total, sawTotal)
		}
		calls = append(calls, done)
	})

	if len(calls) != 51 { // initial 0 + one per chunk
		t.Fatalf("progress called %d times, want 51", len(calls))
	}
	if calls[0] != 0 {
		t.Errorf("first call done = %d, want 0", calls[0])
	}
	max := 0
	for _, d := range calls {
		if d > max {
			max = d
		}
	}
	if max != 50 {
		t.Errorf("max done = %d, want 50", max)
	}
}

func TestEmbedChunks_Empty(t *testing.T) {
	// Must not panic or deadlock on zero chunks.
	embedChunks(embed.NewHashEmbedder(8), nil, nil)
}

func TestEmbedChunks_Single(t *testing.T) {
	// Fewer chunks than workers: the extra workers must exit cleanly.
	emb := embed.NewHashEmbedder(8)
	chunks := []core.Chunk{{Content: "only one"}}
	embedChunks(emb, chunks, nil)
	if !reflect.DeepEqual(chunks[0].Vector, emb.Embed("only one")) {
		t.Fatal("single chunk not embedded correctly")
	}
}
