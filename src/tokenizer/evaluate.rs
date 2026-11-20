//! Paired compression metrics based on actual standard-runtime token IDs.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Result};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use sentencepiece_model::Type;
use serde::Serialize;

use super::{model::Model, records};
use crate::cancel::CancellationToken;
use crate::progress::{ProgressReporter, StageProgress};

#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Ratio {
    Defined { value: f64 },
    Undefined { reason: &'static str },
}

impl Ratio {
    fn of(candidate: u64, baseline: u64) -> Self {
        if baseline == 0 {
            return Self::Undefined {
                reason: "denominator is zero",
            };
        }
        Self::Defined {
            value: candidate as f64 / baseline as f64,
        }
    }
}

impl Default for Ratio {
    fn default() -> Self {
        Self::Undefined {
            reason: "no observations",
        }
    }
}

#[derive(Debug, Default, Serialize)]
pub struct TokenStats {
    pub records_with_byte_fallback: u64,
    pub mean_tokens_per_record: Ratio,
    pub p50_tokens_per_record: Ratio,
    pub p95_tokens_per_record: Ratio,
    pub codepoints_per_token: Ratio,
    pub used_vocab_pieces: usize,
    #[serde(skip)]
    token_lengths: BTreeMap<usize, u64>,
    #[serde(skip)]
    used_ids: HashSet<u32>,
    pub tokens: u64,
    pub unknown_tokens: u64,
    pub byte_tokens: u64,
    pub raw_round_trip_failures: u64,
    pub normalized_round_trip_failures: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct Counts {
    pub text_codepoints: u64,
    pub records: u64,
    pub empty_records: u64,
    pub text_bytes: u64,
    pub improved_records: u64,
    pub unchanged_records: u64,
    pub worsened_records: u64,
    pub baseline: TokenStats,
    pub candidate: TokenStats,
}

#[derive(Debug, Serialize)]
pub struct Metrics {
    pub counts: Counts,
    pub token_count_ratio: Ratio,
    pub percent_tokens_saved: Ratio,
}

impl From<Counts> for Metrics {
    fn from(mut counts: Counts) -> Self {
        counts
            .baseline
            .finalize(counts.records, counts.text_codepoints);
        counts
            .candidate
            .finalize(counts.records, counts.text_codepoints);
        let token_count_ratio = Ratio::of(counts.candidate.tokens, counts.baseline.tokens);
        let percent_tokens_saved = match token_count_ratio {
            Ratio::Defined { value } => Ratio::Defined {
                value: 100.0 * (1.0 - value),
            },
            Ratio::Undefined { reason } => Ratio::Undefined { reason },
        };
        Self {
            counts,
            token_count_ratio,
            percent_tokens_saved,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct FileReport {
    pub input: PathBuf,
    pub metrics: Metrics,
}

#[derive(Debug, Serialize)]
pub struct SourceReport {
    pub source_group: String,
    pub metrics: Metrics,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub baseline_model_blake3: String,
    pub candidate_model_blake3: String,
    pub baseline_vocab_size: usize,
    pub candidate_vocab_size: usize,
    pub same_vocab_size: bool,
    pub normalization_matches: bool,
    pub files: Vec<FileReport>,
    pub sources: Vec<SourceReport>,
    pub total: Metrics,
}

/// Counts both models on identical records without retaining corpus-sized state.
/// Reports reconstruction loss separately from intentional normalization changes.
pub fn compare(
    baseline: &Model,
    candidate: &Model,
    inputs: &[PathBuf],
    max_record_bytes: usize,
    quiet: bool,
    cancellation: &CancellationToken,
) -> Result<Report> {
    let progress = if quiet {
        ProgressReporter::silent()
    } else {
        ProgressReporter::new(false)
    };
    compare_with_options(
        baseline,
        candidate,
        inputs,
        &Options {
            max_record_bytes,
            threads: 1,
            progress: &progress,
        },
        cancellation,
    )
}

pub struct Options<'a> {
    pub max_record_bytes: usize,
    pub threads: usize,
    pub progress: &'a ProgressReporter,
}

pub fn compare_with_options(
    baseline: &Model,
    candidate: &Model,
    inputs: &[PathBuf],
    options: &Options<'_>,
    cancellation: &CancellationToken,
) -> Result<Report> {
    ensure!(!inputs.is_empty(), "at least one input file is required");
    ensure!(baseline.same_normalization(candidate),
        "normalization, denormalization, or user-defined symbols differ; token counts are not comparable");
    ensure!(options.threads > 0, "thread count must be positive");
    let pool = ThreadPoolBuilder::new()
        .num_threads(options.threads)
        .build()?;
    let bytes: u64 = inputs
        .iter()
        .map(|p| p.metadata().map(|m| m.len()))
        .collect::<std::io::Result<Vec<_>>>()?
        .iter()
        .sum();
    let stage = options
        .progress
        .stage_bar("evaluating held-out tokenizer pairs", bytes);
    let mut files = Vec::new();
    let mut sources: BTreeMap<String, Counts> = BTreeMap::new();
    let mut total = Counts::default();
    for path in inputs {
        let counts = compare_file(
            baseline,
            candidate,
            path,
            options.max_record_bytes,
            &pool,
            &stage,
            cancellation,
        )?;
        total.add(&counts);
        sources.entry(source_group(path)).or_default().add(&counts);
        files.push(FileReport {
            input: path.clone(),
            metrics: counts.into(),
        });
    }
    stage.finish(format!("evaluated {} records", total.records));
    Ok(Report {
        baseline_model_blake3: baseline.fingerprint(),
        candidate_model_blake3: candidate.fingerprint(),
        baseline_vocab_size: baseline.proto.pieces.len(),
        candidate_vocab_size: candidate.proto.pieces.len(),
        same_vocab_size: baseline.proto.pieces.len() == candidate.proto.pieces.len(),
        normalization_matches: true,
        files,
        sources: sources
            .into_iter()
            .map(|(source_group, counts)| SourceReport {
                source_group,
                metrics: Metrics::from(counts),
            })
            .collect(),
        total: total.into(),
    })
}

fn compare_file(
    baseline: &Model,
    candidate: &Model,
    path: &Path,
    max_record_bytes: usize,
    pool: &ThreadPool,
    progress: &StageProgress,
    cancellation: &CancellationToken,
) -> Result<Counts> {
    progress.set_message(format!("evaluating {}", path.display()));
    let mut counts = Counts::default();
    records::batches(path, max_record_bytes, cancellation, |batch| {
        let size = batch
            .records
            .len()
            .div_ceil(pool.current_num_threads())
            .max(1);
        let parts: Result<Vec<Counts>> = pool.install(|| {
            batch
                .records
                .par_chunks(size)
                .map(|records| {
                    let mut local = Counts::default();
                    for text in records {
                        cancellation.check()?;
                        local.observe(baseline, candidate, text)?;
                    }
                    Ok(local)
                })
                .collect()
        });
        for part in parts? {
            counts.add(&part);
        }
        progress.inc(batch.bytes);
        Ok(())
    })?;
    Ok(counts)
}

impl TokenStats {
    fn observe(&mut self, model: &Model, text: &str) -> Result<usize> {
        let pieces = model.processor.encode(text)?;
        let ids: Vec<u32> = pieces.iter().map(|p| p.id).collect();
        let decoded = model.processor.decode_piece_ids(&ids)?;
        self.tokens += ids.len() as u64;
        *self.token_lengths.entry(ids.len()).or_default() += 1;
        self.used_ids.extend(ids.iter().copied());
        self.records_with_byte_fallback += u64::from(
            ids.iter()
                .any(|id| model.proto.pieces[*id as usize].r#type() == Type::Byte),
        );
        for id in &ids {
            match model.proto.pieces[*id as usize].r#type() {
                Type::Unknown => self.unknown_tokens += 1,
                Type::Byte => self.byte_tokens += 1,
                _ => {}
            }
        }
        self.raw_round_trip_failures += u64::from(decoded != text);
        self.normalized_round_trip_failures +=
            u64::from(model.processor.normalize(&decoded)? != model.processor.normalize(text)?);
        Ok(ids.len())
    }

    fn add(&mut self, other: &Self) {
        self.records_with_byte_fallback += other.records_with_byte_fallback;
        for (length, frequency) in &other.token_lengths {
            *self.token_lengths.entry(*length).or_default() += frequency;
        }
        self.used_ids.extend(other.used_ids.iter().copied());
        self.tokens += other.tokens;
        self.unknown_tokens += other.unknown_tokens;
        self.byte_tokens += other.byte_tokens;
        self.raw_round_trip_failures += other.raw_round_trip_failures;
        self.normalized_round_trip_failures += other.normalized_round_trip_failures;
    }

    fn finalize(&mut self, records: u64, codepoints: u64) {
        self.mean_tokens_per_record = Ratio::of(self.tokens, records);
        self.p50_tokens_per_record = self.quantile(records, 50);
        self.p95_tokens_per_record = self.quantile(records, 95);
        self.codepoints_per_token = Ratio::of(codepoints, self.tokens);
        self.used_vocab_pieces = self.used_ids.len();
        // File reports retain scalar summaries, not one histogram/set per corpus file.
        self.token_lengths.clear();
        self.used_ids.clear();
        self.used_ids.shrink_to_fit();
    }

    fn quantile(&self, records: u64, percent: u128) -> Ratio {
        if records == 0 {
            return Ratio::default();
        }
        let rank = (u128::from(records) * percent).div_ceil(100);
        let mut seen = 0_u128;
        for (length, frequency) in &self.token_lengths {
            seen += u128::from(*frequency);
            if seen >= rank {
                return Ratio::Defined {
                    value: *length as f64,
                };
            }
        }
        Ratio::default()
    }
}

/// Prepared shards retain a source hint, not original-file provenance.
fn source_group(path: &Path) -> String {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    name.split_once("-source_")
        .and_then(|(_, suffix)| suffix.split_once("-len_"))
        .map_or_else(|| name.to_string(), |(source, _)| source.to_owned())
}

impl Counts {
    fn observe(&mut self, baseline: &Model, candidate: &Model, text: &str) -> Result<()> {
        let before = self.baseline.observe(baseline, text)?;
        let after = self.candidate.observe(candidate, text)?;
        self.records += 1;
        self.empty_records += u64::from(text.is_empty());
        self.text_bytes += text.len() as u64;
        self.text_codepoints += text.chars().count() as u64;
        match after.cmp(&before) {
            std::cmp::Ordering::Less => self.improved_records += 1,
            std::cmp::Ordering::Equal => self.unchanged_records += 1,
            std::cmp::Ordering::Greater => self.worsened_records += 1,
        }
        Ok(())
    }

    fn add(&mut self, other: &Self) {
        self.text_codepoints += other.text_codepoints;
        self.records += other.records;
        self.empty_records += other.empty_records;
        self.text_bytes += other.text_bytes;
        self.improved_records += other.improved_records;
        self.unchanged_records += other.unchanged_records;
        self.worsened_records += other.worsened_records;
        self.baseline.add(&other.baseline);
        self.candidate.add(&other.candidate);
    }
}

#[cfg(test)]
mod metric_tests {
    use super::*;

    #[test]
    fn aggregate_quantiles_and_utilization_use_all_observations() {
        let mut first = Counts {
            records: 3,
            text_codepoints: 30,
            ..Counts::default()
        };
        first.candidate.tokens = 13;
        first.candidate.token_lengths.extend([(1, 2), (11, 1)]);
        first.candidate.used_ids.extend([0, 1]);
        let mut second = Counts {
            records: 2,
            text_codepoints: 10,
            ..Counts::default()
        };
        second.candidate.tokens = 4;
        second.candidate.token_lengths.insert(2, 2);
        second.candidate.used_ids.extend([1, 2]);
        let mut total = Counts::default();
        total.add(&first);
        total.add(&second);
        let metrics = Metrics::from(total);
        let stats = metrics.counts.candidate;
        assert_eq!(stats.mean_tokens_per_record, Ratio::Defined { value: 3.4 });
        assert_eq!(stats.p50_tokens_per_record, Ratio::Defined { value: 2.0 });
        assert_eq!(stats.p95_tokens_per_record, Ratio::Defined { value: 11.0 });
        assert_eq!(
            stats.codepoints_per_token,
            Ratio::Defined { value: 40.0 / 17.0 }
        );
        assert_eq!(stats.used_vocab_pieces, 3);
        assert!(stats.token_lengths.is_empty());
        assert!(stats.used_ids.is_empty());
    }

    #[test]
    fn empty_evaluation_has_undefined_compression_and_quantiles() {
        let metrics = Metrics::from(Counts::default());
        assert!(matches!(metrics.token_count_ratio, Ratio::Undefined { .. }));
        assert!(matches!(
            metrics.counts.candidate.p95_tokens_per_record,
            Ratio::Undefined { .. }
        ));
        assert!(matches!(
            metrics.counts.candidate.codepoints_per_token,
            Ratio::Undefined { .. }
        ));
    }
}
