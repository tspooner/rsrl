//! Function approximation and value function representation module.
use crate::params::Buffer;

#[cfg(test)]
pub(crate) mod mocking;

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct StateUpdate<S, E = f64> {
    pub state: S,
    pub error: E,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct StateActionUpdate<S, A, E = f64> {
    pub state: S,
    pub action: A,
    pub error: E,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct GradientUpdate<J: Buffer>(pub J);

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct ScaledGradientUpdate<J: Buffer> {
    pub alpha: f64,
    pub jacobian: J,
}

pub mod linear;
pub mod tabular;

pub mod transforms;
import_all!(composition);
