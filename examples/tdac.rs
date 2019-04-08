extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::actor_critic::TDAC,
    core::{run, Evaluation, SerialExperiment},
    domains::{ContinuousMountainCar, Domain},
    fa::{basis::{Composable, fixed::Fourier}, LFA},
    logging,
    policies::parameterised::Gaussian1d,
    prediction::td::TD,
};

fn main() {
    let domain = ContinuousMountainCar::default();
    let bases = Fourier::from_space(3, domain.state_space()).with_constant();

    let critic = TD::new(LFA::scalar(bases.clone()), 0.02, 0.99);
    let policy = Gaussian1d::new(
        LFA::scalar(bases),
        0.5,
    );

    let mut agent = TDAC::new(critic, policy, 0.001, 0.99);

    let logger = logging::root(logging::stdout());
    let domain_builder = Box::new(ContinuousMountainCar::default);

    // Training phase:
    let _training_result = {
        // Start a serial learning experiment up to 1000 steps per episode.
        let e = SerialExperiment::new(&mut agent, domain_builder.clone(), 1000);

        // Realise 1000 episodes of the experiment generator.
        run(e, 2000, Some(logger.clone()))
    };

    // Testing phase:
    let testing_result = Evaluation::new(&mut agent, domain_builder).next().unwrap();

    info!(logger, "solution"; testing_result);
}
