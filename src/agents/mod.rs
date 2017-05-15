use geometry::Space;


pub trait Agent<S: Space> {
    fn handle_terminal(&mut self) {}
}

pub use self::control::ControlAgent;
pub use self::prediction::PredictionAgent;


pub mod control;
pub mod prediction;
