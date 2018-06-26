extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::mc::REINFORCE,
    core::{make_shared, run, Evaluation, SerialExperiment},
    domains::{ContinuousMountainCar, Domain},
    fa::{projectors::fixed::Fourier, LFA},
    logging,
    policies::parameterised::Gaussian1d,
};

fn main() {
    let domain = ContinuousMountainCar::default();
    let policy = {
        // Build the linear value function using a fourier basis projection and the
        // appropriate eligibility trace.
        let bases = Fourier::from_space(5, domain.state_space());
        let fa = LFA::simple(bases.clone());

        // Build a stochastic behaviour policy with exponential epsilon.
        Gaussian1d::new(fa, 1.0)
    };

    let mut agent = REINFORCE::new(make_shared(policy), 0.01, 0.99);

    let logger = logging::root(logging::stdout());
    let domain_builder = Box::new(ContinuousMountainCar::default);

    // Training phase:
    let _training_result = {
        // Start a serial learning experiment up to 1000 steps per episode.
        let e = SerialExperiment::new(&mut agent, domain_builder.clone(), 1000);

        // Realise 1000 episodes of the experiment generator.
        run(e, 10000, Some(logger.clone()))
    };

    // Testing phase:
    let testing_result = Evaluation::new(&mut agent, domain_builder).next().unwrap();

    info!(logger, "solution"; testing_result);
}
