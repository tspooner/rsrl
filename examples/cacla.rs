extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::actor_critic::CACLA,
    core::{make_shared, run, Evaluation, SerialExperiment},
    domains::{ContinuousMountainCar, Domain},
    fa::{basis::{Composable, fixed::Fourier}, LFA},
    logging,
    policies::parameterised::{Dirac, Gaussian1d},
    prediction::td::TD,
};

fn main() {
    let domain = ContinuousMountainCar::default();
    let bases = Fourier::from_space(3, domain.state_space()).with_constant();

    let mean_fa = make_shared(LFA::scalar(bases.clone()));
    let target_policy = Dirac::new(mean_fa.clone());
    let behaviour_policy = Gaussian1d::new(mean_fa, 1.0);

    let critic = TD::new(LFA::scalar(bases), 0.1, 1.0);

    let mut agent = CACLA::new(critic, target_policy, behaviour_policy, 0.001, 1.0);

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
