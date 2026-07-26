package core

import (
	"reflect"
	"testing"
)

// The fiddly bit: []float32 must survive a round-trip through the BLOB encoding
// bit-for-bit.
func TestEncodeDecodeVec(t *testing.T) {
	v := []float32{1.5, -2.25, 0, 3.1415927}
	if got := DecodeVec(EncodeVec(v)); !reflect.DeepEqual(got, v) {
		t.Errorf("round-trip = %v, want %v", got, v)
	}
}
