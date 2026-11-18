cargo-run +args:
    RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native" CFLAGS="${CFLAGS:-} -march=native" uv run cargo run --release -- {{args}}

train config="cfg.json":
    just cargo-run train --config {{config}}

train-only config="cfg.json":
    just cargo-run train --config {{config}} --train-only

check:
    uv run cargo fmt --all -- --check
    uv run cargo test
    uv run cargo test python_runtime_compatibility -- --ignored
    uv run cargo clippy --all-targets --all-features -- -D warnings
    RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native" CFLAGS="${CFLAGS:-} -march=native" uv run cargo build --release
    uv run ruff check .
    uv run --with pyrefly pyrefly check

fmt:
    cargo fmt --all
    uv run ruff format .
