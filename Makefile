.PHONY: help fmt check test build clean install clippy fix

help:
	@echo "rvnllm Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  fmt      - Format code"
	@echo "  check    - Check code (no build)"
	@echo "  clippy   - Run clippy lints"
	@echo "  test     - Run tests"
	@echo "  build    - Build debug"
	@echo "  release  - Build release"
	@echo "  fix      - Fix clippy issues automatically"
	@echo "  clean    - Clean build artifacts"
	@echo "  install  - Install binaries"

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features -- -D warnings -A dead_code

fix:
	cargo clippy --all-targets --all-features --fix --allow-dirty -- -A dead_code

check:
	cargo check --all-targets --all-features

test:
	cargo test --all-targets --all-features

build:
	cargo build

release:
	cargo build --release

clean:
	cargo clean

install: release
	cargo install --path .

pre-commit: fmt clippy test

ci: fmt-check clippy test build
