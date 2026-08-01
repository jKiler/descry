package embed

// meanPool collapses a [tokens × dim] hidden-state matrix (flattened
// row-major, as the ONNX output arrives) into one dim-length vector: the
// mean of the rows whose mask is 1. Rows with mask 0 (padding) are ignored.
func meanPool(hidden []float32, mask []int64, dim int) []float32 {
	pooled := make([]float32, dim)
	var kept float32
	for tok, m := range mask {
		if m == 0 {
			continue
		}
		kept++
		row := hidden[tok*dim : (tok+1)*dim]
		for i, x := range row {
			pooled[i] += x
		}
	}
	if kept == 0 {
		return pooled // all padding: zero vector, not NaN
	}
	for i := range pooled {
		pooled[i] /= kept
	}
	return pooled
}

// clsPool takes the first row of the hidden-state matrix — the [CLS] token.
//
// Which pooling a model wants is part of its recipe, not a free choice.
// sentence-transformers models like MiniLM are trained with mean pooling; the
// BGE family is trained with CLS pooling, and scoring one with the other's
// recipe measures the mismatch rather than the model. A comparison between two
// encoders is only informative if each runs the way it was trained.
func clsPool(hidden []float32, dim int) []float32 {
	pooled := make([]float32, dim)
	if len(hidden) < dim {
		return pooled
	}
	copy(pooled, hidden[:dim])
	return pooled
}
