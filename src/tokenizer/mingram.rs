//! SentencePiece-compatible MinGram: hard EM on token-count-dominated paths, then flat pruning.
//! Adapted from https://arxiv.org/abs/2606.27019 and its reference implementation.
//! Native float32 scores approximate the reference's probability tie-break; they do not
//! guarantee lexicographic minimum-token inference for arbitrarily long inputs.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use sentencepiece_model::{ModelType, SentencePiece, Type};
use serde::{Deserialize, Serialize};

use super::{
    checkpoint::{self, Store},
    model::Model,
    records,
};
use crate::cancel::CancellationToken;
use crate::progress::ProgressReporter;

const SCORE_DELTA: f64 = 1.0 / 100_000.0;
const UNUSED_LOG_PROBABILITY: f64 = -100.0;

pub struct Options {
    pub vocab_size: usize,
    pub iterations: usize,
    pub max_record_bytes: usize,
    pub quiet: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PassReport {
    pub phase: String,
    pub records: u64,
    pub tokens: u64,
    pub byte_tokens: u64,
    pub used_pieces: usize,
}

#[derive(Debug, Serialize)]
pub struct IdMapping {
    pub seed_id: usize,
    pub output_id: usize,
    pub piece: String,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub algorithm: &'static str,
    pub score_delta: f64,
    pub seed_model_blake3: String,
    pub output_model_blake3: String,
    pub training_files: Vec<PathBuf>,
    pub max_record_bytes: usize,
    pub seed_vocab_size: usize,
    pub output_vocab_size: usize,
    pub protected_pieces: usize,
    pub passes: Vec<PassReport>,
    pub id_mapping: Vec<IdMapping>,
}

pub struct Trained {
    pub model: Model,
    pub report: Report,
}

#[derive(Serialize, Deserialize)]
struct Usage {
    counts: Vec<u64>,
    report: PassReport,
}

pub struct Runtime<'a> {
    pool: ThreadPool,
    progress: &'a ProgressReporter,
    checkpoints: Option<&'a Store>,
}

impl<'a> Runtime<'a> {
    pub fn new(
        threads: usize,
        progress: &'a ProgressReporter,
        checkpoints: Option<&'a Store>,
    ) -> Result<Self> {
        ensure!(threads > 0, "thread count must be positive");
        Ok(Self {
            pool: ThreadPoolBuilder::new().num_threads(threads).build()?,
            progress,
            checkpoints,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct Snapshot {
    next_file: usize,
    bytes: u64,
    usage: Usage,
}

/// Replays corpus files for each pass, retaining only per-vocabulary counts and one record.
/// Rejects unknown IDs or normalized reconstruction loss instead of optimizing lossy text.
pub fn train(
    seed: &Model,
    inputs: &[PathBuf],
    options: &Options,
    cancellation: &CancellationToken,
) -> Result<Trained> {
    let progress = if options.quiet {
        ProgressReporter::silent()
    } else {
        ProgressReporter::new(false)
    };
    let runtime = Runtime::new(1, &progress, None)?;
    train_with_runtime(seed, inputs, options, &runtime, cancellation)
}

pub fn train_with_runtime(
    seed: &Model,
    inputs: &[PathBuf],
    options: &Options,
    runtime: &Runtime<'_>,
    cancellation: &CancellationToken,
) -> Result<Trained> {
    validate(seed, inputs, options)?;
    let initial = count(seed, inputs, options, "seed", runtime, cancellation)?;
    let mut counts = initial.counts;
    let mut model = reestimate(seed, &counts)?;
    let mut passes = vec![initial.report];
    for iteration in 0..options.iterations {
        let usage = count(
            &model,
            inputs,
            options,
            &format!("hard_em_{}", iteration + 1),
            runtime,
            cancellation,
        )?;
        model = reestimate(&model, &usage.counts)?;
        counts = usage.counts;
        passes.push(usage.report);
    }
    let protected_pieces = model.proto.pieces.iter().filter(|p| protected(p)).count();
    let selected = select(&model, options.vocab_size, &counts)?;
    let id_mapping = selected
        .iter()
        .enumerate()
        .map(|(output_id, (seed_id, piece))| IdMapping {
            seed_id: *seed_id,
            output_id,
            piece: piece.piece().to_owned(),
        })
        .collect();
    model = model.with_pieces(&selected)?;
    let final_usage = count(&model, inputs, options, "final", runtime, cancellation)?;
    passes.push(final_usage.report);
    Ok(Trained {
        report: Report {
            algorithm: "mingram_sentencepiece_v1",
            score_delta: SCORE_DELTA,
            seed_model_blake3: seed.fingerprint(),
            output_model_blake3: model.fingerprint(),
            training_files: inputs.to_vec(),
            max_record_bytes: options.max_record_bytes,
            seed_vocab_size: seed.proto.pieces.len(),
            output_vocab_size: model.proto.pieces.len(),
            protected_pieces,
            passes,
            id_mapping,
        },
        model,
    })
}

fn validate(seed: &Model, inputs: &[PathBuf], options: &Options) -> Result<()> {
    ensure!(!inputs.is_empty(), "at least one training file is required");
    ensure!(options.iterations > 0, "iterations must be positive");
    ensure!(
        options.max_record_bytes > 0,
        "max_record_bytes must be positive"
    );
    ensure!(
        options.vocab_size > 0 && options.vocab_size <= seed.proto.pieces.len(),
        "target vocabulary must be positive and no larger than the seed"
    );
    let spec = seed
        .proto
        .trainer_spec
        .as_ref()
        .context("seed has no trainer specification")?;
    ensure!(
        matches!(spec.model_type(), ModelType::Bpe | ModelType::Unigram),
        "seed must be a BPE or Unigram model"
    );
    ensure!(
        seed.proto.pieces.iter().filter(|p| protected(p)).count() <= options.vocab_size,
        "protected characters and special pieces exceed the target vocabulary"
    );
    for (id, piece) in seed.proto.pieces.iter().enumerate() {
        ensure!(
            piece.r#type() == Type::Normal || id < options.vocab_size,
            "reserved piece ID {id} would lie outside the target vocabulary"
        );
    }
    Ok(())
}

fn protected(piece: &SentencePiece) -> bool {
    piece.r#type() != Type::Normal || piece.piece().chars().count() == 1
}

fn empty_usage(model: &Model, phase: &str) -> Usage {
    Usage {
        counts: vec![0; model.proto.pieces.len()],
        report: PassReport {
            phase: phase.into(),
            records: 0,
            tokens: 0,
            byte_tokens: 0,
            used_pieces: 0,
        },
    }
}

fn merge_usage(total: &mut Usage, part: Usage) {
    for (total, count) in total.counts.iter_mut().zip(part.counts) {
        *total += count;
    }
    total.report.records += part.report.records;
    total.report.tokens += part.report.tokens;
    total.report.byte_tokens += part.report.byte_tokens;
}

fn count_key(
    model: &Model,
    inputs: &[PathBuf],
    options: &Options,
    phase: &str,
    runtime: &Runtime<'_>,
) -> Result<String> {
    checkpoint::key(&(
        1,
        runtime.checkpoints.map(|s| s.identity.as_str()),
        inputs,
        model.fingerprint(),
        options.vocab_size,
        options.iterations,
        options.max_record_bytes,
        phase,
    ))
}

fn count(
    model: &Model,
    inputs: &[PathBuf],
    options: &Options,
    phase: &str,
    runtime: &Runtime<'_>,
    cancellation: &CancellationToken,
) -> Result<Usage> {
    let key = count_key(model, inputs, options, phase, runtime)?;
    let restored = runtime
        .checkpoints
        .map(|store| store.read::<Snapshot>(phase, &key))
        .transpose()?
        .flatten();
    let mut state = restored.unwrap_or_else(|| Snapshot {
        next_file: 0,
        bytes: 0,
        usage: empty_usage(model, phase),
    });
    ensure!(
        state.next_file <= inputs.len()
            && state.usage.counts.len() == model.proto.pieces.len()
            && state.usage.report.phase == phase
            && state.usage.counts.iter().sum::<u64>() == state.usage.report.tokens,
        "invalid MinGram checkpoint counters"
    );
    let sizes: Vec<_> = inputs
        .iter()
        .map(|path| path.metadata().map(|m| m.len()))
        .collect::<std::io::Result<_>>()?;
    ensure!(
        state.bytes == sizes[..state.next_file].iter().sum::<u64>(),
        "checkpoint byte offset differs from corpus"
    );
    let stage = runtime.progress.stage_bar(
        format!(
            "MinGram {phase}: {} threads, {} completed files, {} saved records",
            runtime.pool.current_num_threads(),
            state.next_file,
            state.usage.report.records
        ),
        sizes.iter().sum(),
    );
    stage.inc(state.bytes);
    for (index, path) in inputs.iter().enumerate().skip(state.next_file) {
        cancellation.check()?;
        stage.set_message(format!(
            "MinGram {phase}: file {}/{}, {} records",
            index + 1,
            inputs.len(),
            state.usage.report.records
        ));
        records::batches(path, options.max_record_bytes, cancellation, |batch| {
            let parts = count_batch(model, &batch, runtime, cancellation).with_context(|| {
                format!("counting {} from line {}", path.display(), batch.first_line)
            })?;
            for part in parts {
                merge_usage(&mut state.usage, part);
            }
            stage.inc(batch.bytes);
            state.bytes += batch.bytes;
            Ok(())
        })?;
        state.next_file = index + 1;
        state.usage.report.used_pieces = state.usage.counts.iter().filter(|c| **c > 0).count();
        if let Some(store) = runtime.checkpoints {
            store.write(phase, &key, &state)?;
        }
    }
    ensure!(
        state.usage.report.tokens > 0,
        "training corpus emits no tokens"
    );
    stage.finish(format!(
        "MinGram {phase}: {} records, {} tokens",
        state.usage.report.records, state.usage.report.tokens
    ));
    Ok(state.usage)
}

fn count_batch(
    model: &Model,
    batch: &records::Batch,
    runtime: &Runtime<'_>,
    cancellation: &CancellationToken,
) -> Result<Vec<Usage>> {
    let chunk_size = batch
        .records
        .len()
        .div_ceil(runtime.pool.current_num_threads())
        .max(1);
    runtime.pool.install(|| {
        batch
            .records
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk, records)| {
                let mut usage = empty_usage(model, "batch");
                for (offset, text) in records.iter().enumerate() {
                    cancellation.check()?;
                    observe(model, text, &mut usage.counts, &mut usage.report).with_context(
                        || {
                            format!(
                                "record {}",
                                batch.first_line + (chunk * chunk_size + offset) as u64
                            )
                        },
                    )?;
                }
                Ok(usage)
            })
            .collect()
    })
}

fn observe(model: &Model, text: &str, counts: &mut [u64], report: &mut PassReport) -> Result<()> {
    let pieces = model.processor.encode(text)?;
    let ids: Vec<u32> = pieces.iter().map(|p| p.id).collect();
    for id in &ids {
        let piece = &model.proto.pieces[*id as usize];
        ensure!(
            piece.r#type() != Type::Unknown,
            "training text emits unknown IDs; provide character coverage or byte fallback"
        );
        counts[*id as usize] += 1;
        report.byte_tokens += u64::from(piece.r#type() == Type::Byte);
    }
    let decoded = model.processor.decode_piece_ids(&ids)?;
    ensure!(
        model.processor.normalize(&decoded)? == model.processor.normalize(text)?,
        "training text fails the normalized ID round trip"
    );
    report.records += 1;
    report.tokens += ids.len() as u64;
    Ok(())
}

fn reestimate(model: &Model, counts: &[u64]) -> Result<Model> {
    let total = counts.iter().sum::<u64>() as f64;
    let pieces: Vec<_> = model
        .proto
        .pieces
        .iter()
        .enumerate()
        .map(|(id, piece)| {
            let mut piece = piece.clone();
            if piece.r#type() == Type::Normal {
                let log_probability = if counts[id] == 0 {
                    UNUSED_LOG_PROBABILITY
                } else {
                    (counts[id] as f64 / total).ln()
                };
                piece.score = Some((-1.0 + SCORE_DELTA * log_probability) as f32);
            }
            (id, piece)
        })
        .collect();
    model.with_pieces(&pieces)
}

/// Pins reserved IDs, then fills remaining slots with protected characters and ranked pieces.
fn select(model: &Model, target: usize, counts: &[u64]) -> Result<Vec<(usize, SentencePiece)>> {
    let mut ranked: Vec<_> = model
        .proto
        .pieces
        .iter()
        .enumerate()
        .filter(|(_, p)| p.r#type() == Type::Normal)
        .collect();
    ranked.sort_by(|(a_id, a), (b_id, b)| {
        protected(b)
            .cmp(&protected(a))
            .then_with(|| counts[*b_id].cmp(&counts[*a_id]))
            .then_with(|| a_id.cmp(b_id))
    });
    let reserved = model
        .proto
        .pieces
        .iter()
        .filter(|p| p.r#type() != Type::Normal)
        .count();
    let mut kept: Vec<_> = ranked.into_iter().take(target - reserved).collect();
    kept.sort_by_key(|(id, _)| *id);
    let mut normals = kept.into_iter();
    let mut output = Vec::with_capacity(target);
    for id in 0..target {
        let piece = &model.proto.pieces[id];
        if piece.r#type() != Type::Normal {
            output.push((id, piece.clone()));
        } else {
            let (old_id, piece) = normals
                .next()
                .context("insufficient normal pieces for target vocabulary")?;
            output.push((old_id, piece.clone()));
        }
    }
    Ok(output)
}

/// Writes a model, vocabulary, and training/ID-mapping report without overwriting files.
pub fn save(trained: &Trained, output: &Path) -> Result<()> {
    let vocab = output.with_extension("vocab");
    let report = output.with_extension("mingram.json");
    ensure!(
        output.extension().is_some_and(|ext| ext == "model"),
        "output must end in .model"
    );
    for path in [output, vocab.as_path(), report.as_path()] {
        ensure!(!path.exists(), "refusing to overwrite {}", path.display());
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut vocab_file = BufWriter::new(File::options().write(true).create_new(true).open(&vocab)?);
    for piece in &trained.model.proto.pieces {
        writeln!(vocab_file, "{}\t{}", piece.piece(), piece.score())?;
    }
    vocab_file.flush()?;
    let mut report_file =
        BufWriter::new(File::options().write(true).create_new(true).open(&report)?);
    serde_json::to_writer_pretty(&mut report_file, &trained.report)?;
    report_file.flush()?;
    let mut model_file = File::options().write(true).create_new(true).open(output)?;
    model_file.write_all(trained.model.serialized())?;
    Ok(())
}

/// Publishes stage artifacts atomically; the workflow owns compatibility and completion markers.
pub(crate) fn save_atomic(trained: &Trained, output: &Path) -> Result<()> {
    let mut vocab = Vec::new();
    for piece in &trained.model.proto.pieces {
        writeln!(vocab, "{}\t{}", piece.piece(), piece.score())?;
    }
    checkpoint::atomic_write(&output.with_extension("vocab"), &vocab)?;
    checkpoint::atomic_write(
        &output.with_extension("mingram.json"),
        &serde_json::to_vec_pretty(&trained.report)?,
    )?;
    checkpoint::atomic_write(output, trained.model.serialized())
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    fn options(size: usize) -> Options {
        Options {
            vocab_size: size,
            iterations: 2,
            max_record_bytes: 1024,
            quiet: true,
        }
    }

    #[test]
    fn parallel_and_file_resumed_models_match_sequential_bytes_and_reports() {
        let temp = tempfile::TempDir::new().unwrap();
        let seed =
            crate::tokenizer::tests::model(&["a", "b", "ab", "ก", "้", "ก้", "中", "文", "中文"]);
        let first = temp.path().join("first.txt");
        let second = temp.path().join("second.txt");
        fs::write(&first, "abก้中文\n".repeat(records::BATCH_RECORDS + 3)).unwrap();
        fs::write(&second, "中文abก้\n".repeat(31)).unwrap();
        let inputs = vec![first.clone(), second];
        let options = options(seed.proto.pieces.len() - 1);
        let cancel = CancellationToken::default();
        let progress = ProgressReporter::silent();
        let sequential = train(&seed, &inputs, &options, &cancel).unwrap();
        let parallel = train_with_runtime(
            &seed,
            &inputs,
            &options,
            &Runtime::new(4, &progress, None).unwrap(),
            &cancel,
        )
        .unwrap();
        assert_eq!(sequential.model.serialized(), parallel.model.serialized());
        assert_eq!(
            serde_json::to_value(&sequential.report).unwrap(),
            serde_json::to_value(&parallel.report).unwrap()
        );

        let prefix = count(
            &seed,
            std::slice::from_ref(&first),
            &options,
            "seed",
            &Runtime::new(1, &progress, None).unwrap(),
            &cancel,
        )
        .unwrap();
        let store = Store::new(temp.path().join("checkpoints"), "verified-corpus".into());
        let runtime = Runtime::new(3, &progress, Some(&store)).unwrap();
        let key = count_key(&seed, &inputs, &options, "seed", &runtime).unwrap();
        store
            .write(
                "seed",
                &key,
                &Snapshot {
                    next_file: 1,
                    bytes: first.metadata().unwrap().len(),
                    usage: prefix,
                },
            )
            .unwrap();
        let resumed = train_with_runtime(&seed, &inputs, &options, &runtime, &cancel).unwrap();
        assert_eq!(sequential.model.serialized(), resumed.model.serialized());
        assert_eq!(
            serde_json::to_value(&sequential.report).unwrap(),
            serde_json::to_value(&resumed.report).unwrap()
        );
        let completed = train_with_runtime(
            &seed,
            &inputs,
            &options,
            &Runtime::new(2, &progress, Some(&store)).unwrap(),
            &cancel,
        )
        .unwrap();
        assert_eq!(completed.model.serialized(), resumed.model.serialized());
        let mut incompatible = options;
        incompatible.iterations += 1;
        assert!(train_with_runtime(&seed, &inputs, &incompatible, &runtime, &cancel).is_err());
        let other = Store::new(temp.path().join("checkpoints"), "different-corpus".into());
        assert!(train_with_runtime(
            &seed,
            &inputs,
            &incompatible,
            &Runtime::new(2, &progress, Some(&other)).unwrap(),
            &cancel
        )
        .is_err());
    }
}
