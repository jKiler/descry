// Package index is the orchestrator: walk a directory, chunk each file, embed
// each chunk, and add it to the store. It wires the chunker, embedder, and store
// behind interfaces so any of them can be swapped without touching this code.
package index

import (
	"bufio"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/jKiler/descry/internal/chunk"
	"github.com/jKiler/descry/internal/core"
	"github.com/jKiler/descry/internal/embed"
	"github.com/jKiler/descry/internal/lang"
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

// IndexDir walks root, indexing files whose extension a language pack claims
// (internal/lang owns that list). Hidden dirs and common vendor/build dirs are
// skipped.
func IndexDir(p *Pipeline, root string) (int, error) { return IndexDirRel(p, root, root) }

// IndexDirRel is IndexDir with the stored paths made relative to base rather
// than to the walked directory, so a caller can index a subtree while keeping
// repository-rooted paths. base must be a prefix of root.
func IndexDirRel(p *Pipeline, root, base string) (int, error) {
	report := func(phase string, done, total int) {
		if p.Progress != nil {
			p.Progress(phase, done, total)
		}
	}

	files := 0
	var chunks []core.Chunk
	report(PhaseScanning, 0, 0)
	err := WalkFiles(root, base, func(rel, content string) error {
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
	// measured near-linear, unlike batching (measured slower: padding inflates short chunks).
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

// skipDirs are directories no index should descend into: dependency trees and
// build output, which are neither the user's code nor small.
var skipDirs = map[string]bool{"node_modules": true, "vendor": true, "dist": true}

// WalkFiles calls fn(rel, content) for every file under root that belongs in an
// index — a language pack claims its extension, and it is neither test code nor
// generated. rel is slash-separated and relative to base. Exported so the
// evaluation harness selects exactly the files the indexer would, instead of
// re-deriving the rules and drifting from them.
func WalkFiles(root, base string, fn func(rel, content string) error) error {
	return filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if name != "." && (strings.HasPrefix(name, ".") || skipDirs[name]) {
				return filepath.SkipDir
			}
			return nil
		}
		// Symlinked files are not followed. os.ReadFile would resolve the link, so
		// a link named foo.go pointing outside the repository puts content the
		// caller never offered into their index, and a link to a file already in
		// the tree indexes it twice. Directory links are already ignored: WalkDir
		// reports them as non-directories and never descends.
		if d.Type()&os.ModeSymlink != 0 {
			return nil
		}
		if lang.ForPath(p) == nil {
			return nil
		}
		// A size cap before the read, because everything downstream is unbounded
		// in the file's length: the parse, the cover, and — for a file with no
		// blank lines and no grammar — a single chunk holding all of it. Source
		// files do not reach this; minified bundles, lockfiles and vendored blobs
		// do, and they are not what anyone is searching for.
		if fi, err := d.Info(); err == nil && fi.Size() > maxFileBytes {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return nil // skip unreadable files rather than abort
		}
		rel, _ := filepath.Rel(base, p)
		rel = filepath.ToSlash(rel)
		content := string(data)
		if skipFile(rel, content) {
			return nil
		}
		return fn(rel, content)
	})
}

// maxFileBytes is the largest file the indexer will read. Chosen well above real
// source — the largest file in the Kubernetes tree is under 3 MB — so the cap
// only catches machine-generated bulk.
const maxFileBytes = 4 << 20

// generatedHeadLines is how far into a file the generated-code marker is looked
// for. Every convention (Go's `// Code generated ... DO NOT EDIT.`, the
// `@generated` tag, protoc's banner) puts it in the file's opening comment, so
// a short window finds them all while a stray "DO NOT EDIT" in real code
// further down cannot false-positive.
const generatedHeadLines = 20

// skipFile reports whether a source file should be left out of the index. Test
// files and generated code (large repos can be ~40% generated conversion/
// deepcopy) bloat the index and crowd out real results without being what
// anyone searches for. Both judgments come from the file's language pack
// (internal/lang), so a new language brings its own conventions with it.
func skipFile(path, content string) bool {
	if lang.IsTest(path) {
		return true
	}
	return lang.IsGenerated(path, head(content, generatedHeadLines))
}

// head returns a file's leading comment block — the blank and comment lines
// before the first line of actual code, capped at n lines.
//
// The cap alone would not be enough. Every generated-code convention puts its
// marker in the opening banner, so that is the only place worth looking; a
// "DO NOT EDIT" *in* the code (a warning on a hand-written constant, say) must
// not condemn the file. The original Go-only rule expressed this as "before the
// package clause"; this is the same rule without needing to know the language's
// keyword for it.
func head(s string, n int) string {
	var b strings.Builder
	sc := bufio.NewScanner(strings.NewReader(s))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for i := 0; i < n && sc.Scan(); i++ {
		line := strings.TrimSpace(sc.Text())
		if line != "" && !isCommentLine(line) {
			break
		}
		b.WriteString(line)
		b.WriteByte('\n')
	}
	return b.String()
}

// commentStarts are the line prefixes that open or continue a comment in the
// languages descry indexes. Recognizing them by prefix keeps head language-
// agnostic; a false negative only means a generated file gets indexed.
var commentStarts = []string{"//", "#", "/*", "*/", "*", "--", ";", "<!--"}

func isCommentLine(line string) bool {
	for _, p := range commentStarts {
		if strings.HasPrefix(line, p) {
			return true
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

// embedText is what the embedder actually sees for a chunk.
//
// A chunker that builds an enriched representation (language, module path,
// qualified symbol, signature — see core.Chunk.EmbedText) supplies it directly.
// Otherwise the chunk gets the original treatment: the relative file path as a
// header line, then the content. The path names the component in words the
// model understands after WordPiece ("internal/search/bm25.go"), so queries
// naming a file or subsystem land nearer its chunks — a measured MRR gain
// (README "Retrieval"). Changing either form changes stored vectors, so bump
// pipelineVersion when you do.
func embedText(c core.Chunk) string {
	if c.EmbedText != "" {
		return c.EmbedText
	}
	if c.Path == "" {
		return c.Content
	}
	return c.Path + "\n" + c.Content
}
