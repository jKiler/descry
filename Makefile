# descry — developer tasks.
#
# Embedding runs on ONNX Runtime, which onnxruntime_go dlopens at runtime. So a
# build needs cgo (a C compiler), but NOT the onnxruntime library — descry
# downloads and caches that on first use. Set DESCRY_ORT_LIB to use your own.

BINARY := descry
PKG    := ./cmd/descry

.DEFAULT_GOAL := help

.PHONY: help build install test vet fmt fmt-check tidy check clean skill skill-check

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

build: ## Build the CLI
	go build -o $(BINARY) $(PKG)

install: ## Install the CLI into GOBIN
	go install $(PKG)

test: ## Run all tests
	go test ./...

vet: ## Run go vet
	go vet ./...

fmt: ## Format all Go source
	gofmt -w .

fmt-check: ## Fail if any Go source is unformatted
	@out=$$(gofmt -l .); if [ -n "$$out" ]; then echo "unformatted:"; echo "$$out"; exit 1; fi

tidy: ## Tidy go.mod / go.sum
	go mod tidy

skill: ## Regenerate skills/descry/SKILL.md from internal/skill
	go run ./cmd/genskill

skill-check: ## Fail if the committed skill drifted from internal/skill
	go run ./cmd/genskill --check

check: fmt-check vet skill-check test ## Run fmt-check, vet, skill-check, and tests (CI gate)

clean: ## Remove build artifacts
	rm -f $(BINARY)
	go clean
