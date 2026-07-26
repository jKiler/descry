package search

import (
	"encoding/binary"
	"hash/fnv"
	"math"

	"github.com/jKiler/descry/internal/core"
)

// bm25Format is the on-disk layout version for Encode/DecodeBM25. Bump it on any
// change to the byte layout so an old blob is rejected (rebuilt) rather than
// misread.
const bm25Format = 1

// Encode serializes the inverted index to a compact binary blob (varint
// postings) so it can be persisted and loaded instead of rebuilt. The chunk
// slice itself is NOT stored — only a hash of the chunk ids, which DecodeBM25
// checks against the chunks it is given.
func (b *BM25) Encode() []byte {
	buf := make([]byte, 0, 1<<20)
	buf = append(buf, bm25Format)
	buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(b.K1))
	buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(b.B))
	pt := byte(0)
	if b.PathTokens {
		pt = 1
	}
	buf = append(buf, pt)
	buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(b.avgLen))

	buf = binary.AppendUvarint(buf, uint64(len(b.chunks)))
	buf = binary.LittleEndian.AppendUint64(buf, chunkIDHash(b.chunks))

	for _, dl := range b.docLen {
		buf = binary.AppendUvarint(buf, uint64(dl))
	}

	buf = binary.AppendUvarint(buf, uint64(len(b.postings)))
	for term, pl := range b.postings {
		buf = binary.AppendUvarint(buf, uint64(len(term)))
		buf = append(buf, term...)
		buf = binary.AppendUvarint(buf, uint64(len(pl)))
		for _, p := range pl {
			buf = binary.AppendUvarint(buf, uint64(p.doc))
			buf = binary.AppendUvarint(buf, uint64(p.tf))
		}
	}
	return buf
}

// DecodeBM25 rebuilds a BM25 from Encode's blob, reattaching the given chunks.
// It returns ok=false — so the caller rebuilds — if the blob is the wrong
// format, truncated, or was built for a different set of chunks (count or id
// hash mismatch). The chunks must be in the same order they were indexed in.
func DecodeBM25(data []byte, chunks []core.Chunk) (*BM25, bool) {
	d := &decoder{buf: data}
	if d.u8() != bm25Format {
		return nil, false
	}
	b := &BM25{}
	b.K1 = math.Float64frombits(d.u64())
	b.B = math.Float64frombits(d.u64())
	b.PathTokens = d.u8() == 1
	b.avgLen = math.Float64frombits(d.u64())

	n := int(d.uvarint())
	idHash := d.u64()
	if d.err || n != len(chunks) || idHash != chunkIDHash(chunks) {
		return nil, false
	}
	b.chunks = chunks

	b.docLen = make([]int, n)
	for i := range b.docLen {
		b.docLen[i] = int(d.uvarint())
	}

	numTerms := int(d.uvarint())
	b.postings = make(map[string][]posting, numTerms)
	for t := 0; t < numTerms && !d.err; t++ {
		term := string(d.bytes(int(d.uvarint())))
		pl := make([]posting, int(d.uvarint()))
		for i := range pl {
			pl[i] = posting{doc: uint32(d.uvarint()), tf: uint32(d.uvarint())}
		}
		b.postings[term] = pl
	}
	if d.err {
		return nil, false
	}
	return b, true
}

// chunkIDHash is an order-sensitive fingerprint of the chunk ids, so a blob is
// only accepted for the exact chunk set (and order) it was built from.
func chunkIDHash(chunks []core.Chunk) uint64 {
	h := fnv.New64a()
	for _, c := range chunks {
		_, _ = h.Write([]byte(c.ID))
		_, _ = h.Write([]byte{0})
	}
	return h.Sum64()
}

// decoder is a bounds-checked cursor over a byte slice; any read past the end
// sets err, which every read then short-circuits on.
type decoder struct {
	buf []byte
	pos int
	err bool
}

func (d *decoder) u8() byte {
	if d.err || d.pos >= len(d.buf) {
		d.err = true
		return 0
	}
	v := d.buf[d.pos]
	d.pos++
	return v
}

func (d *decoder) u64() uint64 {
	if d.err || d.pos+8 > len(d.buf) {
		d.err = true
		return 0
	}
	v := binary.LittleEndian.Uint64(d.buf[d.pos:])
	d.pos += 8
	return v
}

func (d *decoder) uvarint() uint64 {
	if d.err {
		return 0
	}
	v, k := binary.Uvarint(d.buf[d.pos:])
	if k <= 0 {
		d.err = true
		return 0
	}
	d.pos += k
	return v
}

func (d *decoder) bytes(n int) []byte {
	if d.err || n < 0 || d.pos+n > len(d.buf) {
		d.err = true
		return nil
	}
	v := d.buf[d.pos : d.pos+n]
	d.pos += n
	return v
}
