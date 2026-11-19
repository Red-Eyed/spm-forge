export RUSTFLAGS := ""
export CARGO_ENCODED_RUSTFLAGS := ""
export CFLAGS := ""
export CXXFLAGS := ""
export CARGO_TARGET_DIR := "target/hpc-portable"

cargo-run +args:
    uv run cargo run --release -- {{args}}

train config="cfg.json":
    just cargo-run train --config {{config}}

train-only config="cfg.json":
    just cargo-run train --config {{config}} --train-only

check:
    uv run cargo fmt --all -- --check
    uv run cargo test
    uv run cargo test python_runtime_compatibility -- --ignored
    uv run cargo test mingram_config_workflow -- --ignored
    uv run cargo clippy --all-targets --all-features -- -D warnings
    uv run cargo build --release
    uv run ruff check .
    uv run pyrefly check --search-path demos/mingram

fmt:
    cargo fmt --all
    uv run ruff format .
