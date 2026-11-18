use std::io::{IsTerminal, Read};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};

const STAGE_BAR_TEMPLATE: &str =
    "{bar:40.cyan/blue} {bytes}/{total_bytes} {elapsed_precise} eta {eta_precise} {msg}";
const PLAIN_STATUS_INTERVAL: Duration = Duration::from_secs(10);

pub struct ProgressReporter {
    json_output: bool,
    quiet: bool,
}

impl ProgressReporter {
    pub fn new(json_output: bool) -> Self {
        Self {
            json_output,
            quiet: false,
        }
    }

    pub fn silent() -> Self {
        Self {
            json_output: false,
            quiet: true,
        }
    }

    pub fn stage(&self, message: impl Into<String>) -> StageProgress {
        if self.quiet {
            return StageProgress::interactive(ProgressBar::hidden());
        }
        let message = message.into();
        if self.json_output || !std::io::stderr().is_terminal() {
            eprintln!("{message}");
            return StageProgress::plain(message);
        }

        let bar = ProgressBar::new_spinner();
        bar.set_style(
            ProgressStyle::with_template("{spinner} {msg}")
                .expect("spinner progress template is static"),
        );
        bar.enable_steady_tick(std::time::Duration::from_millis(120));
        bar.set_message(message);
        StageProgress::interactive(bar)
    }

    pub fn stage_bar(&self, message: impl Into<String>, total: u64) -> StageProgress {
        if self.quiet {
            return StageProgress::interactive(ProgressBar::hidden());
        }
        let message = message.into();
        if self.json_output || !std::io::stderr().is_terminal() {
            eprintln!("{message}");
            let stage = StageProgress::plain(message);
            stage.bar.set_length(total);
            return stage;
        }

        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::with_template(STAGE_BAR_TEMPLATE)
                .expect("bar progress template is static")
                .progress_chars("=> "),
        );
        bar.enable_steady_tick(Duration::from_millis(250));
        bar.set_message(message);
        StageProgress::interactive(bar)
    }
}

pub struct StageProgress {
    bar: ProgressBar,
    plain_output: bool,
    started: Instant,
    last_plain_emit_seconds: AtomicU64,
}

impl StageProgress {
    fn interactive(bar: ProgressBar) -> Self {
        Self {
            bar,
            plain_output: false,
            started: Instant::now(),
            last_plain_emit_seconds: AtomicU64::new(0),
        }
    }

    fn plain(initial_message: String) -> Self {
        Self {
            bar: ProgressBar::hidden(),
            plain_output: true,
            started: Instant::now(),
            last_plain_emit_seconds: AtomicU64::new(0),
        }
        .with_initial_message(initial_message)
    }

    fn with_initial_message(self, initial_message: String) -> Self {
        self.bar.set_message(initial_message);
        self
    }
}

impl StageProgress {
    pub fn set_message(&self, message: impl Into<String>) {
        let message = message.into();
        self.bar.set_message(message.clone());
        self.emit_plain_status(message);
    }

    pub fn inc(&self, delta: u64) {
        self.bar.inc(delta);
        if self.plain_output
            && self
                .started
                .elapsed()
                .as_secs()
                .saturating_sub(self.last_plain_emit_seconds.load(Ordering::Relaxed))
                >= PLAIN_STATUS_INTERVAL.as_secs()
        {
            self.emit_plain_status(self.bar.message().to_string());
        }
    }

    pub fn reader<R: Read>(&self, inner: R) -> ProgressReader<'_, R> {
        ProgressReader {
            inner,
            progress: self,
        }
    }

    pub fn finish(self, message: impl Into<String>) {
        let message = message.into();
        if self.plain_output {
            eprintln!("{message}");
        }
        self.bar.finish_with_message(message);
    }

    fn emit_plain_status(&self, message: String) {
        if !self.plain_output {
            return;
        }

        let seconds = self.started.elapsed().as_secs();
        let last = self.last_plain_emit_seconds.load(Ordering::Relaxed);
        if seconds.saturating_sub(last) < PLAIN_STATUS_INTERVAL.as_secs() {
            return;
        }
        if self
            .last_plain_emit_seconds
            .compare_exchange(last, seconds, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        if let Some(total) = self.bar.length() {
            let position = self.bar.position();
            let percent = if total == 0 {
                100.0
            } else {
                100.0 * position as f64 / total as f64
            };
            eprintln!("{message}: {position}/{total} bytes ({percent:.1}%), {seconds}s elapsed");
        } else {
            eprintln!("{message}: {seconds}s elapsed");
        }
    }
}

pub struct ProgressReader<'a, R> {
    inner: R,
    progress: &'a StageProgress,
}

impl<R: Read> Read for ProgressReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let bytes = self.inner.read(buffer)?;
        self.progress.inc(bytes as u64);
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_bar_tracks_position() {
        let progress = ProgressReporter::new(false);
        let stage = progress.stage_bar("fixing corpus", 3);

        assert_eq!(stage.bar.length(), Some(3));
        assert!(STAGE_BAR_TEMPLATE.contains("{bytes}/{total_bytes}"));
        assert!(STAGE_BAR_TEMPLATE.contains("eta {eta_precise}"));
        stage.inc(1);
        assert_eq!(stage.bar.position(), 1);
        stage.finish("done");
    }
}
