//! Temporal-difference control algorithms.
// Off-policy:
pub mod q_learning;
pub mod q_lambda;
pub mod q_sigma;
pub mod pal;
pub mod greedy_gq;

pub use self::{
    q_learning::QLearning,
    q_lambda::QLambda,
    q_sigma::QSigma,
    pal::PAL,

    greedy_gq::GreedyGQ,
};

// On-policy:
pub mod sarsa;
pub mod sarsa_lambda;
pub mod expected_sarsa;

pub use self::{
    sarsa::SARSA,
    sarsa_lambda::SARSALambda,
    expected_sarsa::ExpectedSARSA,
};

// TODO:
// PQ(lambda) - http://proceedings.mlr.press/v32/sutton14.pdf
