// Package index is the orchestrator: walk a directory, chunk each file, embed
// each chunk, and add it to the store. It wires the chunker, embedder, and store
// behind interfaces so any of them can be swapped without touching this code.
package index

import (
	"bufio"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/jKiler/descry/internal/chunk"
	"github.com/jKiler/descry/internal/core"
	"github.com/jKiler/descry/internal/embed"
	"github.com/jKiler/descry/internal/store"
)

// Pipeline wires the three building blocks behind interfaces, so any of them
// can be swapped (simple -> real) without touching this code.
type Pipeline struct {
	Chunker chunk.Chunker
	Emb     embed.Embedder
	Store   store.Store

	// Progress, if non-nil, observes indexing: a phase (see the Phase constants)
	// plus a done/total count, where total is 0 when the size isn't known yet.
	// On a large repository every phase takes real time, so all of them report —
	// otherwise the command looks hung while it walks the tree or commits.
	//
	// During PhaseEmbedding it is called from the embed worker goroutines, so
	// implementations must be safe for concurrent use, and counts may arrive
	// slightly out of order.
	Progress func(phase string, done, total int)
}

// Indexing phases reported through Pipeline.Progress.
const (
	PhaseScanning  = "scanning"  // walking the tree, reading and chunking files
	PhaseEmbedding = "embedding" // vectorizing the chunks (usually the long pole)
	PhaseStoring   = "storing"   // committing chunks + vectors to the index
)

// New returns a pipeline with the given components.
func New(c chunk.Chunker, e embed.Embedder, s store.Store) *Pipeline {
	return &Pipeline{Chunker: c, Emb: e, Store: s}
}

// defaultExts are the file extensions descry indexes.
var defaultExts = map[string]bool{
	".go": true, ".ts": true, ".js": true, ".py": true, ".rs": true,
	".java": true, ".md": true, ".txt": true, ".json": true, ".yaml": true,
}

// IndexDir walks root, indexing files with known extensions. Hidden dirs and
// common vendor/build dirs are skipped.
func IndexDir(p *Pipeline, root string) (int, error) {
	report := func(phase string, done, total int) {
		if p.Progress != nil {
			p.Progress(phase, done, total)
		}
	}

	files := 0
	var chunks []core.Chunk
	report(PhaseScanning, 0, 0)
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if name != "." && (strings.HasPrefix(name, ".") ||
				name == "node_modules" || name == "vendor" || name == "dist") {
				return filepath.SkipDir
			}
			return nil
		}
		if !defaultExts[strings.ToLower(filepath.Ext(path))] {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil // skip unreadable files rather than abort
		}
		rel, _ := filepath.Rel(root, path)
		content := string(data)
		if skipFile(rel, content) {
			return nil
		}
		chunks = append(chunks, p.Chunker.Chunk(rel, content)...)
		files++
		// The tree walk is silent otherwise, and on a large repository it runs
		// for a long time before embedding starts. Total is unknown until the
		// walk finishes, so report the running count.
		report(PhaseScanning, files, 0)
		return nil
	})
	if err != nil {
		return files, err
	}
	// The walk is done, so the count is final: report it as a completed phase.
	report(PhaseScanning, files, files)

	// Embedding usually dominates indexing time, so fan it out across CPU cores —
	// measured near-linear, unlike batching (see DESIGN.md).
	embedChunks(p.Emb, chunks, func(done, total int) { report(PhaseEmbedding, done, total) })

	// One transaction for the whole index, instead of a commit per chunk. This is
	// a single opaque call, so report only its start and finish — but on a large
	// repository it is many seconds of silence otherwise.
	report(PhaseStoring, 0, len(chunks))
	if err := p.Store.AddBatch(chunks); err != nil {
		return files, err
	}
	report(PhaseStoring, len(chunks), len(chunks))
	return files, nil
}

// generatedMarker is Go's standard generated-code header
// (https://golang.org/s/generatedcode). It must appear on its own line before
// the package clause.
var generatedMarker = regexp.MustCompile(`^// Code generated .* DO NOT EDIT\.$`)

// skipFile reports whether a source file should be left out of the index. Test
// files and generated code (large repos can be ~40% generated conversion/
// deepcopy) bloat the index and crowd out real results without being what
// anyone searches for.
func skipFile(path, content string) bool {
	if strings.HasSuffix(path, "_test.go") {
		return true
	}
	// A generated file carries the marker before its package clause; scan only
	// that far so a stray "DO NOT EDIT" in real code doesn't false-positive.
	sc := bufio.NewScanner(strings.NewReader(content))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if generatedMarker.MatchString(line) {
			return true
		}
		if strings.HasPrefix(line, "package ") {
			break
		}
	}
	return false
}

// embedChunks fills each chunk's Vector using a pool of GOMAXPROCS workers that
// pull chunk indices off a shared atomic counter (self-balancing, since chunk
// lengths — and thus embed times — vary). Writes are by index, so the result is
// identical to a serial loop regardless of worker count or scheduling.
//
// Requires embed.Embedder.Embed to be safe for concurrent use — both the hash
// and ONNX embedders are (the latter verified under -race).
// progress, if non-nil, is invoked as documented on Pipeline.Progress.
func embedChunks(emb embed.Embedder, chunks []core.Chunk, progress func(done, total int)) {
	n := len(chunks)
	if n == 0 {
		return
	}
	if progress != nil {
		progress(0, n)
	}
	workers := min(runtime.GOMAXPROCS(0), n)

	var next, done atomic.Int64
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= n {
					return
				}
				chunks[i].Vector = emb.Embed(embedText(chunks[i]))
				if progress != nil {
					progress(int(done.Add(1)), n)
				}
			}
		}()
	}
	wg.Wait()
}

// embedText is what the embedder actually sees for a chunk: the relative file
// path as a header line, then the content. The path names the component in
// words the model understands after WordPiece ("internal/search/bm25.go"), so
// queries that name a file or subsystem land nearer its chunks — a measured MRR
// gain (see DESIGN.md). Changing this changes stored vectors, so bump
// pipelineVersion when you do.
func embedText(c core.Chunk) string {
	if c.Path == "" {
		return c.Content
	}
	return c.Path + "\n" + c.Content
}
