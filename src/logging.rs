extern crate time;


use log;
use log::{LogRecord, LogLevel, LogMetadata, SetLoggerError, LogLevelFilter};

pub struct DefaultLogger;

impl DefaultLogger {
    pub fn init() -> Result<(), SetLoggerError> {
        log::set_logger(|max_log_level| {
            max_log_level.set(LogLevelFilter::Info);
            Box::new(DefaultLogger)
        })
    }
}

impl log::Log for DefaultLogger {
    fn enabled(&self, metadata: &LogMetadata) -> bool {
        metadata.level() <= LogLevel::Info
    }

    fn log(&self, record: &LogRecord) {
        if self.enabled(record.metadata()) {
            println!("[{}] {}", time::now().ctime(), record.args());
        }
    }
}
