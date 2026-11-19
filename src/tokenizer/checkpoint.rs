//! Versioned, checksummed snapshots written by one owning training process.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    version: u32,
    key: String,
    payload: String,
    blake3: String,
}

pub struct Store {
    directory: PathBuf,
    pub identity: String,
}

impl Store {
    pub fn new(directory: PathBuf, identity: String) -> Self {
        Self {
            directory,
            identity,
        }
    }

    pub fn read<T: DeserializeOwned>(&self, name: &str, key: &str) -> Result<Option<T>> {
        let path = self.directory.join(format!("{name}.json"));
        if !path.exists() {
            return Ok(None);
        }
        let envelope: Envelope = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("invalid checkpoint {}", path.display()))?;
        ensure!(
            envelope.version == 1 && envelope.key == key,
            "checkpoint settings or inputs differ: {}; use a new model_prefix",
            path.display()
        );
        ensure!(
            blake3::hash(envelope.payload.as_bytes()).to_hex().as_str() == envelope.blake3,
            "checkpoint checksum mismatch: {}",
            path.display()
        );
        Ok(Some(serde_json::from_str(&envelope.payload)?))
    }

    pub fn write<T: Serialize>(&self, name: &str, key: &str, value: &T) -> Result<()> {
        let payload = serde_json::to_string(value)?;
        let envelope = Envelope {
            version: 1,
            key: key.into(),
            blake3: blake3::hash(payload.as_bytes()).to_hex().to_string(),
            payload,
        };
        atomic_write(
            &self.directory.join(format!("{name}.json")),
            &serde_json::to_vec(&envelope)?,
        )
    }
}

#[derive(Serialize, Deserialize)]
pub struct Artifacts {
    files: Vec<(PathBuf, String)>,
}

impl Artifacts {
    pub fn capture(paths: &[PathBuf]) -> Result<Self> {
        let files = paths
            .iter()
            .map(|p| Ok((p.clone(), blake3::hash(&fs::read(p)?).to_hex().to_string())))
            .collect::<Result<_>>()?;
        Ok(Self { files })
    }

    pub fn verify(&self, expected: &[PathBuf]) -> Result<()> {
        ensure!(
            self.files.iter().map(|(p, _)| p).eq(expected.iter()),
            "checkpoint artifact paths differ"
        );
        for (path, hash) in &self.files {
            ensure!(
                blake3::hash(
                    &fs::read(path)
                        .with_context(|| format!("reading saved artifact {}", path.display()))?
                )
                .to_hex()
                .as_str()
                    == hash,
                "saved artifact changed: {}",
                path.display()
            );
        }
        Ok(())
    }
}

pub fn key(value: &impl Serialize) -> Result<String> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

/// The rename publishes a complete file; abandoned pending files never count as checkpoints.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let pending = path.with_extension(format!(
        "{}.pending",
        path.extension().unwrap_or_default().to_string_lossy()
    ));
    let mut file = File::create(&pending)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&pending, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatches_and_corruption_and_ignores_pending_writes() {
        let temp = tempfile::TempDir::new().unwrap();
        let store = Store::new(temp.path().into(), "corpus".into());
        fs::write(temp.path().join("counts.json.pending"), "interrupted").unwrap();
        assert_eq!(store.read::<Vec<u64>>("counts", "a").unwrap(), None);
        store.write("counts", "a", &vec![1_u64, 2]).unwrap();
        assert_eq!(
            store.read::<Vec<u64>>("counts", "a").unwrap(),
            Some(vec![1, 2])
        );
        assert!(store.read::<Vec<u64>>("counts", "b").is_err());
        let path = temp.path().join("counts.json");
        let mut envelope: Envelope = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        envelope.payload = "[100,200]".into();
        fs::write(path, serde_json::to_vec(&envelope).unwrap()).unwrap();
        assert!(store.read::<Vec<u64>>("counts", "a").is_err());
    }
}
