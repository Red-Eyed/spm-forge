//! Bounded, replayable UTF-8 line input shared by training and evaluation.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::cancel::CancellationToken;

pub const DEFAULT_MAX_RECORD_BYTES: usize = 1024 * 1024;

/// Visits every record, removing only LF or CRLF record delimiters.
/// Oversized records and invalid UTF-8 are errors, never silently skipped.
pub fn visit(
    path: &Path,
    max_record_bytes: usize,
    cancellation: &CancellationToken,
    mut consume: impl FnMut(&str, u64) -> Result<()>,
) -> Result<()> {
    ensure!(max_record_bytes > 0, "max_record_bytes must be positive");
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut bytes = Vec::new();
    let mut line = 0_u64;
    loop {
        cancellation.check()?;
        bytes.clear();
        let read = reader
            .by_ref()
            .take(max_record_bytes as u64 + 3)
            .read_until(b'\n', &mut bytes)?;
        if read == 0 {
            return Ok(());
        }
        line += 1;
        if bytes.last() == Some(&b'\n') {
            bytes.pop();
            if bytes.last() == Some(&b'\r') {
                bytes.pop();
            }
        }
        ensure!(
            bytes.len() <= max_record_bytes,
            "{}:{line}: record exceeds {max_record_bytes} bytes",
            path.display()
        );
        let text = std::str::from_utf8(&bytes)
            .with_context(|| format!("{}:{line}: invalid UTF-8", path.display()))?;
        consume(text, read as u64).with_context(|| format!("{}:{line}", path.display()))?;
    }
}

pub const BATCH_RECORDS: usize = 4096;
pub const BATCH_BYTES: usize = 4 * 1024 * 1024;

pub struct Batch {
    pub records: Vec<String>,
    pub bytes: u64,
    pub first_line: u64,
}

/// Retains at most one batch plus one size-limited record, independent of corpus size.
pub fn batches(
    path: &Path,
    max_record_bytes: usize,
    cancellation: &CancellationToken,
    mut consume: impl FnMut(Batch) -> Result<()>,
) -> Result<()> {
    let mut batch = Batch {
        records: Vec::new(),
        bytes: 0,
        first_line: 1,
    };
    let mut line = 0;
    visit(path, max_record_bytes, cancellation, |text, bytes| {
        line += 1;
        batch.records.push(text.to_owned());
        batch.bytes += bytes;
        if batch.records.len() >= BATCH_RECORDS || batch.bytes >= BATCH_BYTES as u64 {
            let next = Batch {
                records: Vec::new(),
                bytes: 0,
                first_line: line + 1,
            };
            consume(std::mem::replace(&mut batch, next))?;
        }
        Ok(())
    })?;
    if !batch.records.is_empty() {
        consume(batch)?;
    }
    Ok(())
}
