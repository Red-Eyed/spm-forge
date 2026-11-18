# MinGram demonstration log

Implementation evaluated: `4211799` (standard SentencePiece export and Rust hard EM).

The motivating question was whether the new trainer can demonstrate useful compression,
and whether an end user can reproduce that evidence from a fresh clone.

The experiment fixes a shared vocabulary budget of 512 pieces, one authored synthetic
eight-language corpus, and disjoint held-out records. It compares standard BPE and Unigram
against the Rust MinGram adaptation, using actual standard-runtime token IDs and exact
reconstruction checks. The committed generator, input text, SHA-256 manifest, script lock,
Markdown template, decision report, and JSON snapshots make the experiment reviewable.

The locked SentencePiece 0.2.3 wheel run produced 39,206 BPE tokens, 56,344 Unigram tokens,
and 31,429 MinGram tokens on 3,200 held-out records. MinGram saved 19.84% versus BPE and
44.22% versus Unigram overall, with zero unknown IDs or raw reconstruction failures.
Chinese regressed by 68.34% versus Unigram despite improving by 13.01% versus BPE.
The decision report surfaces this regression instead of recommending unconditional adoption.

An earlier exploratory run used the project's Git-built Python SentencePiece rather than
the locked wheel and emitted five additional Unigram tokens in the Japanese group.
The committed report uses the locked wheel result; the exploratory run is not mixed into it.

The corpus has a small, repetitive template grammar. Natural-text compression, dialect
coverage, behavior at 8,192 pieces, downstream model quality, latency, and statistical
robustness remain unproven. The next evaluation should use representative untouched data.

Fresh-clone verification used a separate local checkout of `392aea8`, with no existing
model artifacts or build directory. `uv run --locked demos/mingram/run.py --quiet` built
the binary, verified the committed text, trained the models, and generated the report.

Configuration integration adds `sentencepiece.model_type: "mingram"` to the ordinary
training workflow, with optional positive `mingram.iterations` defaulting to 2. A tagged
algorithm value keeps MinGram options out of the BPE configuration. The Python bridge
receives BPE seed requests only. Synthetic end-to-end testing verifies the config-driven
seed, Rust optimizer, final model, ID mapping, and training reports.
