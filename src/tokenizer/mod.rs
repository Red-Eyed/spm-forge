//! General-purpose tokenizer optimization and paired corpus evaluation.

pub mod evaluate;
pub mod mingram;
pub mod model;
pub(crate) mod records;
pub mod report;
mod wire;

#[cfg(test)]
mod tests;
