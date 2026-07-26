//go:build cgo

package main

// cgoEnabled records how this binary was built. Embedding dlopens ONNX Runtime
// through cgo, so a CGO_ENABLED=0 build compiles and installs cleanly and then
// fails at the first embed — worth reporting up front rather than at first use.
const cgoEnabled = true
