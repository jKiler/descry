package store

import "math"

// defaultRerankMult is how many candidates per requested result the int8
// pre-filter keeps for exact float32 reranking. The pool is topK*mult; higher
// means closer to an exhaustive search at more cost. 6 preserves the exact
// top-K in practice while scanning in int8.
const defaultRerankMult = 6

// quantize maps an L2-normalized vector (components in [-1,1]) to int8 by scaling
// by 127. Clamped to [-127,127] (not -128) so the range stays symmetric.
func quantize(v []float32) []int8 {
	out := make([]int8, len(v))
	quantizeInto(out, v)
	return out
}

func quantizeInto(dst []int8, v []float32) {
	for j, x := range v {
		q := int32(math.Round(float64(x) * 127))
		dst[j] = int8(min(max(q, -127), 127))
	}
}

// dotI8 is the integer dot product of two equal-length int8 rows. Each product
// is at most 127*127, so a 384-dim (or larger) sum stays well within int32.
func dotI8(a, b []int8) int32 {
	var sum int32
	for j, x := range a {
		sum += int32(x) * int32(b[j])
	}
	return sum
}
