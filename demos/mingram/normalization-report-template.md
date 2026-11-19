# NFC versus NFD: decision report

**Recommendation:** {decision} This is not a universal result.

## Controlled comparison

Train equal-sized BPE, Unigram, and MinGram models separately on NFC and NFD copies of the same committed synthetic corpus: 16,000 training records, 3,200 disjoint held-out records, eight language groups, and **512 total pieces per final model**. MinGram uses a 589-piece BPE seed and two hard-EM iterations. Training uses full character coverage, no byte fallback, preserved whitespace, no input sampling, one baseline-training thread, and a 16-code-point maximum piece length.

The comparison is total emitted IDs for canonically equivalent text, not tokens per byte or per code point. Positive percentages below mean NFD is worse for compression.

| Tokenizer | NFC tokens | NFD tokens | NFD increase |
|---|---:|---:|---:|
{totals}

## MinGram by synthetic language group

| Group | NFC tokens | NFD tokens | NFD increase |
|---|---:|---:|---:|
{sources}

## Interpretation

The training alphabet shrinks from **{alphabet_nfc} code points with NFC** to **{alphabet_nfd} with NFD**. A smaller alphabet therefore does not, by itself, imply fewer output tokens. NFD decomposes Korean syllables into Jamo and decomposes accents and some other letters. Shared vocabulary allocation also changes: a group's token count can change even when normalization leaves its own text unchanged.

A fixed code-point piece limit can disadvantage decomposed representations. The same comparison is repeated with a 64-code-point limit as a control. {control} These are fresh training runs; small differences in tied vocabulary candidates can also affect results.

## Reconstruction and deployment boundary

Exact reconstruction of the **corresponding normalized inputs**, with zero unknown IDs across all models and controls: **{lossless}**.

NFC changes **{changed_nfc}** original held-out records; NFD changes **{changed_nfd}**. Exact reconstruction after normalization must not be called exact preservation of the original code-point sequence.

Normalization is performed externally with Python's Unicode database **{unicode_version}**, before both training and evaluation. SentencePiece models use identity normalization and do **not** embed NFC or NFD. Deploying these experimental models on arbitrary unnormalized text would not reproduce this workflow. This comparison does not add production NFC/NFD support or claim that a finite mapping table implements complete Unicode NFC.

## Reproduce

From the repository root:

```sh
uv run --locked demos/mingram/compare_normalization.py
```

Committed text and its SHA-256 manifest are verified first. The runner supplies locked Python packages and CMake, builds the Rust executable, normalizes copies, checks train/test separation, trains all variants, and runs the Rust optimizer and evaluator. Fresh artifacts, normalized text, models, JSON evidence, and this report are written beneath `target/mingram-normalization/`.

The original corpus, word inventories, and templates are committed under `demos/mingram/data/` and `generate.py`. No text is downloaded. Rust/Cargo, a C++ compiler, and uv are prerequisites.

## Decision limits

This is the same small template-generated corpus as the initial demo, with some unnatural combinations. It tests representation and vocabulary effects, not natural-text generalization, dialect coverage, downstream accuracy, or inference latency. Only one initialization per condition is used; this is not a statistical robustness study. The vocabulary budget of 8,192 is not evaluated. Use representative held-out text before choosing a production normalization policy.

Full structured evidence is in [normalization comparison JSON](normalization-results.json), also written beside a freshly generated report. Native per-file evaluator reports are retained under each run's normalization/limit directory.
