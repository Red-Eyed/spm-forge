# Changelog

## 0.6.0

### Highlights

- Train native sparse Unigram tokenizers through the normal configuration and held-out
  reporting workflow, using prebuilt SentencePiece fork wheels.
- Balance multilingual training exposure by text-derived Unicode script, with one shared,
  resumable sample for the candidate tokenizer and matched BPE baseline.
- Inspect code points per token for every held-out shard and source group, plus per-script
  training-sample exposure.

### Tracked Regressions

- Fixed character coverage can require more characters and reserved pieces than fit in an
  8K multilingual vocabulary, especially after balancing. Increase the vocabulary budget
  or lower `character_coverage` if the trainer reports this condition. Automatic character
  coverage in sparse Unigram can instead increase byte fallback and regress individual
  scripts. Neither sparse pruning nor balancing guarantees improved compression.
- Very short `common` records can consume a substantial share of the script-balanced record
  budget. Review `reports/script_sample.json`; set `balancing.enabled` to `false` to restore
  the original sampling policy. The sample cap counts records, not code points.

### Backwards Incompatible Changes

#### Multilingual training defaults

`cfg.json`, `cfg.mingram.json`, and the multilingual presets now enable
`balancing.mode: "script_sample"`. The sampler uses fixed square-root code-point allocation,
translates this to intact-record quotas, and caps quotas at available data without
oversampling. All Latin text shares a group; mixed-script and common-only records have
separate groups. Filenames, language hints, and source labels do not control allocation.
Legacy OCR presets retain conservative balancing.

Both candidate and baseline consume the same saved sample with native resampling disabled.
The default cap is 1,000,000 records. To retain the previous behavior, explicitly set
`balancing.enabled` to `false`. A changed training sample can change vocabulary IDs and
segmentation; downstream models may need adaptation.

Previously prepared unbalanced corpora can be reused with `train-only` when other preparation
settings match. Choose a new `output.model_prefix` when changing trained-model inputs.
Previously resampled conservative corpora require fresh preparation to recover the full
training distribution. Interrupted sample files repeat; completed input-file checkpoints
and sample parts are verified and reused. Interrupted native fitting restarts that fit.

#### SentencePiece wheel source

The Python trainer changes from PyPI SentencePiece `0.2.3` to fork version `0.2.3.post1`
from the explicit GitHub Release wheel index. Run `uv sync --locked` after updating. The
lockfile pins wheel URLs and SHA-256 hashes; SentencePiece source-build fallback remains
disabled. Wheels support CPython 3.11–3.14 including free-threaded 3.14, Linux x86_64 with
glibc 2.27+, and macOS ARM64 with macOS 11+. Unsupported combinations fail installation.
The Rust runtime retains its existing version and native build.

### New Features

- Add `cfg.sparse.json` and `multilingual_sparse_unigram` with `model_type: "unigram"`,
  `use_sparse_pruning`, and `auto_character_coverage` options.
- Add `cfg.bpe.json` and a standalone `multilingual_bpe` preset with script sampling,
  8K pieces, byte fallback, NFC preparation, and fixed character coverage.
- Add bounded-memory, parallel-batch script classification and streaming sample selection,
  with progress reporting, deterministic replay for a fixed ordered corpus and seed,
  and checkpoints after each completed input file.
- Report actual input/selected records and code points by script in Markdown and JSON.
  Aggregate source-group compression using summed code points and tokens rather than
  averages of shard ratios. Source names remain provenance hints, not sampling weights.

### Bug Fixes

- Allow fresh model prefixes to share an existing prepared work directory without being
  blocked by another model's shared native request metadata; untracked artifacts remain
  protected against overwrite.
- Reject incompatible MinGram options when selecting native Unigram.
- Print the `trainer_output.json` path in failure messages so native stderr and Python
  tracebacks can be found directly.

## 0.5.0

### Highlights

- Train multilingual MinGram tokenizers through the normal configuration workflow and
  export standard SentencePiece models.
- Compare token compression and reconstruction on a saved held-out split, with automatic
  Markdown and JSON decision reports.
- Resume interrupted MinGram counting from completed input files and reuse verified seeds,
  models, and completed evaluation.

### Backwards Incompatible Changes

#### Default configuration and prepared data

The default profile changes from `ocr_multilingual` to `multilingual_mingram`: an 8,192-piece
MinGram vocabulary with byte fallback, NFC preparation, and no corpus balancing. Select
`ocr_multilingual` explicitly to retain the legacy BPE/OCR preset settings. A different
tokenizer can change token IDs and segmentation; adapt downstream models before using it.

All configured training runs now reserve approximately 5% of canonical original records for
testing before chunking. Exact canonical duplicates share a partition, and all chunks of a
parent record stay together. Balancing and model fitting use only training data. A split
with an empty partition fails explicitly. Configure `evaluation.test_fraction` and
`evaluation.split_seed` when necessary; this split also applies to legacy BPE presets.

`train-only` requires `reports/split.json` and verifies corpus checksums and preparation
settings. Older prepared runs without this manifest need one preparation run in a new work
directory. Compatible runs with a saved split can adopt an existing completed BPE seed;
MinGram work performed by versions without count checkpoints cannot be recovered.

#### Preset catalog filename

Rename custom references to `cfg.json.ocr` to `cfg.presets.json`. The CLI defaults and bundled
references use the new filename. `cfg.json` and `cfg.mingram.json` retain their names.

#### Python installation

The Python trainer now uses the published SentencePiece `0.2.3` wheel instead of a moving
upstream Git source. Source-build fallback is disabled for that package, so installation
requires a compatible wheel. The Rust binding retains its existing native runtime; this
release does not align the two native versions or introduce another native build.

### New Features

- Add standalone `mingram` and `evaluate` commands, including parallel execution through
  `--num-threads` and optional quiet or JSON output.
- Report paired token counts and savings, code points per token, mean/P50/P95 tokens per
  prepared record, byte fallback, unknown IDs, round-trip failures, vocabulary usage, and
  test-shard results. Token compression does not establish downstream model quality.
- Save versioned, checksummed MinGram counters after each completed corpus file. Atomic
  artifact writes and completion markers distinguish interrupted exports from completed
  models. Changed semantic settings or corpus/model fingerprints reject stale checkpoints.
  Thread count may change when resuming.

### Improvements

- Show byte progress for copying, hashing, MinGram passes, and held-out evaluation. Unknown
  totals use status/spinner output, and non-interactive HPC logs receive periodic stderr
  progress without contaminating JSON stdout.
- Resume with `just train-only cfg.mingram.json` using the same work directory and model
  prefix. An unfinished MinGram file is repeated; unfinished native fitting or evaluation
  restarts that stage. Fixing and balancing must already have completed.
- Clear inherited Rust and C/C++ flags in `just` recipes and use an isolated
  `target/hpc-portable/` build cache to avoid reusing earlier CPU-specific build helpers.

### Performance

- Move script classification into repair workers, cache shard filenames, and keep counters
  beside buffered writers instead of searching shard summaries for every record.
- Parallelize MinGram counting and paired evaluation with bounded batches and worker-local
  integer reductions. Retain all training records and per-record reconstruction checks.
- Stream unbalanced corpus assembly instead of retaining entire shards in memory. Legacy
  explicitly enabled balancing retains its existing in-memory sampling path.

### Documentation

- Present `spm-forge` as a general multilingual tokenizer toolkit, with OCR as one use case.
- Add reproducible multilingual compression and NFC/NFD demos with committed synthetic
  inputs, generated Markdown reports, and documented limitations.
- Record local throughput measurements and byte-equivalence/resume checks in the README
  and MinGram experiment log. HPC throughput and downstream accuracy remain unmeasured.

### Developers

- Forbid unsafe Rust in the application crate and add parallel, checkpoint, interrupted
  export, byte-fallback, and stock Python runtime interoperability checks.
