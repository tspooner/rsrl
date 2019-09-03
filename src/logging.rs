//! Logging utilities module.
use slog::*;
use slog_async;
use slog_term;
use std::{fmt::Debug, fs::File};

pub fn stdout() -> Fuse<slog_async::Async> {
    let decorator = slog_term::TermDecorator::new().build();
    let drain = slog_term::FullFormat::new(decorator).build().fuse();

    slog_async::Async::new(drain)
        .overflow_strategy(slog_async::OverflowStrategy::Block)
        .build()
        .fuse()
}

pub fn file(file: File) -> Fuse<slog_async::Async> {
    let decorator = slog_term::PlainDecorator::new(file);
    let drain = slog_term::FullFormat::new(decorator).build().fuse();

    slog_async::Async::new(drain)
        .overflow_strategy(slog_async::OverflowStrategy::Block)
        .build()
        .fuse()
}

pub fn combine<D1, D2>(drain1: D1, drain2: D2) -> Fuse<Duplicate<D1, D2>>
where
    D1: Drain,
    D1::Ok: Debug,
    D1::Err: Debug,
    D2: Drain,
    D2::Ok: Debug,
    D2::Err: Debug,
{
    Duplicate::new(drain1, drain2).fuse()
}

pub fn root<D: 'static>(drain: D) -> Logger
where D: SendSyncUnwindSafeDrain<Err = Never, Ok = ()>
        + SendSyncRefUnwindSafeDrain<Ok = (), Err = Never> {
    Logger::root(drain, o!())
}
