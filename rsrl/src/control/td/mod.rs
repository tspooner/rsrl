//! Temporal-difference control algorithms.
// Off-policy:
pub mod greedy_gq;
pub mod pal;
pub mod q_lambda;
pub mod q_learning;
pub mod q_sigma;

pub use self::{
    greedy_gq::GreedyGQ,
    pal::PAL,

    q_lambda::QLambda,
    q_learning::QLearning,
    q_sigma::QSigma,
};

// On-policy:
pub mod expected_sarsa;
pub mod sarsa;
pub mod sarsa_lambda;

pub use self::{expected_sarsa::ExpectedSARSA, sarsa::SARSA, sarsa_lambda::SARSALambda};

// TODO:
// PQ(lambda) - http://proceedings.mlr.press/v32/sutton14.pdf
