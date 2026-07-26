package core

import (
	"encoding/binary"
	"math"
)

// EncodeVec serializes a []float32 as raw little-endian bytes (4 per component).
// This is the shared on-disk vector layout: the chunk store and the embed cache
// both persist vectors in this form.
func EncodeVec(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// DecodeVec is the inverse of EncodeVec.
func DecodeVec(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}
