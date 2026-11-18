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
    let mut reader = BufReader::new(file);
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
