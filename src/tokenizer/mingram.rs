//! SentencePiece-compatible MinGram: hard EM on token-count-dominated paths, then flat pruning.
//! Adapted from https://arxiv.org/abs/2606.27019 and its reference implementation.
//! Native float32 scores approximate the reference's probability tie-break; they do not
//! guarantee lexicographic minimum-token inference for arbitrarily long inputs.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use indicatif::ProgressBar;
use sentencepiece_model::{ModelType, SentencePiece, Type};
use serde::Serialize;

use super::{model::Model, records};
use crate::cancel::CancellationToken;

const SCORE_DELTA: f64 = 1.0 / 100_000.0;
const UNUSED_LOG_PROBABILITY: f64 = -100.0;

pub struct Options {
    pub vocab_size: usize,
    pub iterations: usize,
    pub max_record_bytes: usize,
    pub quiet: bool,
}

#[derive(Debug, Serialize)]
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

struct Usage {
    counts: Vec<u64>,
    report: PassReport,
}

/// Replays corpus files for each pass, retaining only per-vocabulary counts and one record.
/// Rejects unknown IDs or normalized reconstruction loss instead of optimizing lossy text.
pub fn train(
    seed: &Model,
    inputs: &[PathBuf],
    options: &Options,
    cancellation: &CancellationToken,
) -> Result<Trained> {
    validate(seed, inputs, options)?;
    let initial = count(seed, inputs, options, "seed", cancellation)?;
    let mut counts = initial.counts;
    let mut model = reestimate(seed, &counts)?;
    let mut passes = vec![initial.report];
    for iteration in 0..options.iterations {
        let usage = count(
            &model,
            inputs,
            options,
            &format!("hard_em_{}", iteration + 1),
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
    let final_usage = count(&model, inputs, options, "final", cancellation)?;
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

fn count(
    model: &Model,
    inputs: &[PathBuf],
    options: &Options,
    phase: &str,
    cancellation: &CancellationToken,
) -> Result<Usage> {
    let mut counts = vec![0_u64; model.proto.pieces.len()];
    let mut report = PassReport {
        phase: phase.to_owned(),
        records: 0,
        tokens: 0,
        byte_tokens: 0,
        used_pieces: 0,
    };
    for path in inputs {
        let progress = if options.quiet {
            ProgressBar::hidden()
        } else {
            ProgressBar::new(path.metadata()?.len())
        };
        progress.set_message(format!("MinGram {phase}: {}", path.display()));
        records::visit(
            path,
            options.max_record_bytes,
            cancellation,
            |text, bytes| {
                observe(model, text, &mut counts, &mut report)?;
                progress.inc(bytes);
                Ok(())
            },
        )?;
        progress.finish_and_clear();
    }
    ensure!(report.tokens > 0, "training corpus emits no tokens");
    report.used_pieces = counts.iter().filter(|c| **c > 0).count();
    Ok(Usage { counts, report })
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
