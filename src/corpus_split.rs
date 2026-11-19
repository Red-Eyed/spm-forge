//! Stable record partitioning and checksummed manifests for replayable held-out evaluation.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::cancel::CancellationToken;
use crate::config::EvaluationConfig;
use crate::progress::{ProgressReporter, StageProgress};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Partition {
    Train,
    Test,
}

/// Assign before chunking; equal canonical records always receive the same partition.
pub fn assign(text: &str, config: &EvaluationConfig) -> Partition {
    let mut hash = blake3::Hasher::new();
    hash.update(&config.split_seed.to_le_bytes());
    hash.update(text.as_bytes());
    let value = u64::from_le_bytes(
        hash.finalize().as_bytes()[..8]
            .try_into()
            .expect("hash has 32 bytes"),
    );
    if (value as f64 / (u64::MAX as f64 + 1.0)) < config.test_fraction {
        Partition::Test
    } else {
        Partition::Train
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CorpusFile {
    pub path: PathBuf,
    pub blake3: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SplitManifest {
    pub version: u32,
    pub policy: EvaluationConfig,
    pub preparation_blake3: String,
    pub training_records_before_balancing: u64,
    pub test_records: u64,
    pub train: Vec<CorpusFile>,
    pub test: Vec<CorpusFile>,
}

pub struct SourceRecords {
    pub training: u64,
    pub test: u64,
}

impl SplitManifest {
    pub fn create(
        train: &[PathBuf],
        test: &[PathBuf],
        policy: EvaluationConfig,
        preparation_blake3: String,
        records: SourceRecords,
        progress: &ProgressReporter,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        ensure!(!train.is_empty() && !test.is_empty(), "train/test split is empty on one side; use more distinct records or adjust evaluation.test_fraction");
        validate_disjoint(train.iter().chain(test.iter()).map(PathBuf::as_path))?;
        let total = file_bytes(train.iter().chain(test))?;
        let stage = progress.stage_bar("hashing prepared train/test corpus", total);
        let manifest = Self {
            version: 1,
            policy,
            preparation_blake3,
            training_records_before_balancing: records.training,
            test_records: records.test,
            train: fingerprint(train, &stage, cancellation)?,
            test: fingerprint(test, &stage, cancellation)?,
        };
        stage.finish("prepared corpus checksums saved");
        Ok(manifest)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        crate::tokenizer::checkpoint::atomic_write(path, &serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }

    pub fn load(
        path: &Path,
        policy: &EvaluationConfig,
        preparation_blake3: &str,
        progress: &ProgressReporter,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        let bytes =
            fs::read(path).context("saved split is missing; run training without --train-only")?;
        let manifest: Self = serde_json::from_slice(&bytes)?;
        ensure!(manifest.version == 1 && &manifest.policy == policy && manifest.preparation_blake3 == preparation_blake3,
            "split or preparation settings changed; rerun without --train-only in a new work directory");
        ensure!(
            !manifest.train.is_empty() && !manifest.test.is_empty(),
            "saved split must contain training and test files"
        );
        validate_disjoint(
            manifest
                .train
                .iter()
                .chain(&manifest.test)
                .map(|file| file.path.as_path()),
        )?;
        let total = file_bytes(manifest.train.iter().chain(&manifest.test).map(|f| &f.path))?;
        let stage = progress.stage_bar("verifying saved train/test corpus", total);
        for file in manifest.train.iter().chain(&manifest.test) {
            stage.set_message(format!("verifying {}", file.path.display()));
            cancellation.check()?;
            ensure!(
                digest(&file.path, &stage, cancellation)? == file.blake3,
                "saved corpus changed: {}",
                file.path.display()
            );
        }
        stage.finish("saved train/test corpus verified");
        Ok(manifest)
    }

    pub fn train_paths(&self) -> Vec<PathBuf> {
        self.train.iter().map(|f| f.path.clone()).collect()
    }
    pub fn test_paths(&self) -> Vec<PathBuf> {
        self.test.iter().map(|f| f.path.clone()).collect()
    }
}

fn validate_disjoint<'a>(paths: impl Iterator<Item = &'a Path>) -> Result<()> {
    let mut seen = HashSet::new();
    for path in paths {
        let canonical = path.canonicalize()?;
        ensure!(
            seen.insert(canonical),
            "duplicate or overlapping corpus path: {}",
            path.display()
        );
    }
    Ok(())
}

fn file_bytes<'a>(paths: impl Iterator<Item = &'a PathBuf>) -> Result<u64> {
    paths
        .map(|p| p.metadata().map(|m| m.len()))
        .collect::<std::io::Result<Vec<_>>>()
        .map(|sizes| sizes.iter().sum())
        .map_err(Into::into)
}

fn fingerprint(
    paths: &[PathBuf],
    progress: &StageProgress,
    cancellation: &CancellationToken,
) -> Result<Vec<CorpusFile>> {
    paths
        .iter()
        .map(|path| {
            cancellation.check()?;
            progress.set_message(format!("hashing {}", path.display()));
            Ok(CorpusFile {
                path: path.clone(),
                blake3: digest(path, progress, cancellation)?,
            })
        })
        .collect()
}

fn digest(
    path: &Path,
    progress: &StageProgress,
    cancellation: &CancellationToken,
) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hash = blake3::Hasher::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        cancellation.check()?;
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
        progress.inc(read as u64);
    }
    Ok(hash.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_normalization::UnicodeNormalization;

    #[test]
    fn rejects_overlapping_partition_files() {
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("same.txt");
        fs::write(&path, "same").unwrap();
        assert!(SplitManifest::create(
            std::slice::from_ref(&path),
            std::slice::from_ref(&path),
            EvaluationConfig::default(),
            "settings".into(),
            SourceRecords {
                training: 1,
                test: 1
            },
            &ProgressReporter::silent(),
            &CancellationToken::default(),
        )
        .is_err());
    }

    #[test]
    fn canonical_duplicates_and_order_have_stable_assignments() {
        let config = EvaluationConfig::default();
        let a: String = "café".nfc().collect();
        let b: String = "cafe\u{0301}".nfc().collect();
        assert_eq!(assign(&a, &config), assign(&b, &config));
        let inputs = ["alpha", "中文", "ก้", "café"];
        let before: Vec<_> = inputs.iter().map(|text| assign(text, &config)).collect();
        let mut after: Vec<_> = inputs
            .iter()
            .rev()
            .map(|text| assign(text, &config))
            .collect();
        after.reverse();
        assert_eq!(before, after);
    }

    #[test]
    fn saved_split_rejects_changed_files() {
        let temp = tempfile::TempDir::new().unwrap();
        let train = temp.path().join("train.txt");
        let test = temp.path().join("test.txt");
        fs::write(&train, "a").unwrap();
        fs::write(&test, "b").unwrap();
        let policy = EvaluationConfig::default();
        let manifest = SplitManifest::create(
            &[train],
            std::slice::from_ref(&test),
            policy.clone(),
            "settings".into(),
            SourceRecords {
                training: 1,
                test: 1,
            },
            &ProgressReporter::silent(),
            &CancellationToken::default(),
        )
        .unwrap();
        let path = temp.path().join("split.json");
        manifest.save(&path).unwrap();
        assert!(SplitManifest::load(
            &path,
            &policy,
            "changed-settings",
            &ProgressReporter::silent(),
            &CancellationToken::default()
        )
        .is_err());
        let mut changed_policy = policy.clone();
        changed_policy.split_seed += 1;
        assert!(SplitManifest::load(
            &path,
            &changed_policy,
            "settings",
            &ProgressReporter::silent(),
            &CancellationToken::default()
        )
        .is_err());
        assert!(SplitManifest::load(
            &path,
            &policy,
            "settings",
            &ProgressReporter::silent(),
            &CancellationToken::default()
        )
        .is_ok());
        fs::write(test, "changed").unwrap();
        assert!(SplitManifest::load(
            &path,
            &policy,
            "settings",
            &ProgressReporter::silent(),
            &CancellationToken::default()
        )
        .is_err());
    }
}
