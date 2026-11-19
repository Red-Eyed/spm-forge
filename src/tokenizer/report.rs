//! Decision-oriented Markdown and structured reports for held-out tokenizer evaluation.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;

use super::evaluate::{Metrics, Ratio, Report};
use crate::config::EffectiveConfig;
use crate::corpus_split::SplitManifest;

#[derive(Serialize)]
struct Evidence<'a> {
    config: &'a EffectiveConfig,
    split: &'a SplitManifest,
    baseline_model: &'a Path,
    candidate_model: &'a Path,
    evaluation: &'a Report,
}

pub fn write(
    config: &EffectiveConfig,
    split: &SplitManifest,
    baseline: &Path,
    candidate: &Path,
    evaluation: &Report,
) -> Result<PathBuf> {
    let directory = config.output.work_dir.join("reports");
    fs::create_dir_all(&directory)?;
    let evidence = Evidence {
        config,
        split,
        baseline_model: baseline,
        candidate_model: candidate,
        evaluation,
    };
    fs::write(
        directory.join("tokenizer.json"),
        serde_json::to_vec_pretty(&evidence)?,
    )?;
    let path = directory.join("tokenizer.md");
    fs::write(&path, markdown(&evidence))?;
    Ok(path)
}

fn markdown(evidence: &Evidence<'_>) -> String {
    [
        summary(evidence),
        compression(evidence.evaluation),
        shards(evidence.evaluation),
        sources(evidence.evaluation),
        interpretation(evidence),
    ]
    .concat()
}

fn summary(evidence: &Evidence<'_>) -> String {
    let report = evidence.evaluation;
    let counts = &report.total.counts;
    let valid = report.same_vocab_size
        && counts.baseline.unknown_tokens == 0
        && counts.candidate.unknown_tokens == 0
        && counts.baseline.raw_round_trip_failures == 0
        && counts.candidate.raw_round_trip_failures == 0
        && counts.baseline.normalized_round_trip_failures == 0
        && counts.candidate.normalized_round_trip_failures == 0
        && counts.records > 0;
    let mut text = format!(
        "# Held-out tokenizer report\n\nCandidate: `{}`. Equal-budget BPE baseline: `{}`.\n\n",
        evidence.candidate_model.display(),
        evidence.baseline_model.display()
    );
    text.push_str(&format!("**Evaluation checks: {}.** Candidate token savings versus BPE: **{}%**. Positive savings are better.\n\n",
        if valid { "PASS" } else { "REVIEW REQUIRED: vocabulary size or reconstruction differs" }, value(&report.total.percent_tokens_saved)));
    text.push_str(&format!("Training records before optional balancing: {}. Held-out original records: {}. Evaluated chunks: {}.\n\n",
        evidence.split.training_records_before_balancing, evidence.split.test_records, counts.records));
    text.push_str(&format!("Split: {:.2}% test, seed {}. Content hashes assign canonical records before chunking. Exact duplicate canonical records and their chunks share a partition. Similar records and common substrings in distinct records are not deduplicated.\n\n",
        100.0 * evidence.split.policy.test_fraction, evidence.split.policy.split_seed));
    text
}

fn compression(report: &Report) -> String {
    let counts = &report.total.counts;
    let mut text = String::new();
    text.push_str("## Compression and coverage\n\n| Metric | BPE | Candidate |\n|---|---:|---:|\n");
    text.push_str(&format!(
        "| Vocabulary pieces (including byte and special pieces) | {} | {} |\n",
        report.baseline_vocab_size, report.candidate_vocab_size
    ));
    let a = &counts.baseline;
    let b = &counts.candidate;
    for (label, left, right) in [
        ("Emitted tokens", a.tokens.to_string(), b.tokens.to_string()),
        (
            "Code points per token (higher is better compression)",
            value(&a.codepoints_per_token),
            value(&b.codepoints_per_token),
        ),
        (
            "Mean tokens per record",
            value(&a.mean_tokens_per_record),
            value(&b.mean_tokens_per_record),
        ),
        (
            "P50 tokens per record",
            value(&a.p50_tokens_per_record),
            value(&b.p50_tokens_per_record),
        ),
        (
            "P95 tokens per record",
            value(&a.p95_tokens_per_record),
            value(&b.p95_tokens_per_record),
        ),
        (
            "Byte-fallback tokens",
            a.byte_tokens.to_string(),
            b.byte_tokens.to_string(),
        ),
        (
            "Records using byte fallback",
            a.records_with_byte_fallback.to_string(),
            b.records_with_byte_fallback.to_string(),
        ),
        (
            "Unknown IDs",
            a.unknown_tokens.to_string(),
            b.unknown_tokens.to_string(),
        ),
        (
            "Raw ID round-trip failures on prepared test text",
            a.raw_round_trip_failures.to_string(),
            b.raw_round_trip_failures.to_string(),
        ),
        (
            "Normalized round-trip failures",
            a.normalized_round_trip_failures.to_string(),
            b.normalized_round_trip_failures.to_string(),
        ),
        (
            "Vocabulary pieces used on test",
            a.used_vocab_pieces.to_string(),
            b.used_vocab_pieces.to_string(),
        ),
    ] {
        text.push_str(&format!("| {label} | {left} | {right} |\n"));
    }
    text.push_str(&format!(
        "\nPrepared code points: {}. Prepared UTF-8 text bytes: {}. Empty test records: {}.\n\n",
        counts.text_codepoints, counts.text_bytes, counts.empty_records
    ));
    text.push_str(&format!("Paired records: {} improved, {} unchanged, {} worsened. Vocabulary utilization is measured on this test set; unused control tokens are normal.\n\n", counts.improved_records, counts.unchanged_records, counts.worsened_records));
    text
}

fn shards(report: &Report) -> String {
    let mut text = String::new();
    text.push_str("## Test-shard breakdown\n\nShards retain the existing source/script hints; no language labels or weights are required. Negative savings are regressions.\n\n| Test shard | Records | BPE tokens | Candidate tokens | Saved | BPE code points/token | Candidate code points/token | Fallback records |\n|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for file in &report.files {
        shard_row(&mut text, &file.input, &file.metrics);
    }
    text
}

fn sources(report: &Report) -> String {
    let mut text = String::from("\n## Source-group summary\n\nGroups use the source hints retained in prepared shard names, not exact original-file identities. Ratios use summed code points divided by summed tokens, rather than averages of shard ratios.\n\n| Source group | Records | BPE tokens | Candidate tokens | Saved | BPE code points/token | Candidate code points/token | Fallback records |\n|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for source in &report.sources {
        shard_row(&mut text, Path::new(&source.source_group), &source.metrics);
    }
    text
}

fn interpretation(evidence: &Evidence<'_>) -> String {
    let report = evidence.evaluation;
    let mut text = String::new();
    text.push_str("\n## Interpretation and reproducibility\n\nToken savings are measured on the saved held-out partition, not the training corpus. Byte tokens count individually. Code points per token is a character-to-token compression metric, not a guarantee of comparable linguistic information across scripts. UTF-8 byte width is not used as the primary quality score.\n\n");
    text.push_str("NFC preparation changes code-point sequences when necessary. Round trips here refer to prepared test text, not the untouched raw source. The exported model retains its SentencePiece normalization settings; corpus-side NFC is not automatically embedded as full Unicode NFC.\n\n");
    text.push_str(&format!("BPE input sampling limit: {} records. Baseline and seed training use only the training partition; native sampling is governed by the recorded configuration. No test record is used to fit either model. This is one training run, not a statistical robustness study or a downstream accuracy/latency benchmark.\n\n", evidence.config.sentencepiece.input_sentence_size));
    text.push_str(&format!("Candidate BLAKE3: `{}`. Baseline BLAKE3: `{}`.\n\nSee [structured metrics](tokenizer.json), [saved partition](split.json), and `effective_config.json` in the work directory.\n", report.candidate_model_blake3, report.baseline_model_blake3));
    text
}

fn shard_row(text: &mut String, path: &Path, metrics: &Metrics) {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .replace('|', "\\|");
    let counts = &metrics.counts;
    text.push_str(&format!(
        "| {name} | {} | {} | {} | {}% | {} | {} | {} |\n",
        counts.records,
        counts.baseline.tokens,
        counts.candidate.tokens,
        value(&metrics.percent_tokens_saved),
        value(&counts.baseline.codepoints_per_token),
        value(&counts.candidate.codepoints_per_token),
        counts.candidate.records_with_byte_fallback
    ));
}

fn value(metric: &Ratio) -> String {
    match metric {
        Ratio::Defined { value } => format!("{value:.3}"),
        Ratio::Undefined { reason } => format!("undefined ({reason})"),
    }
}
