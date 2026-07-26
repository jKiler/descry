//go:build !cgo

package main

// See cgo_on.go: this build cannot embed, and doctor says so.
const cgoEnabled = false
