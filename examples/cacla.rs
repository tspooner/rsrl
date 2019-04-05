extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::actor_critic::CACLA,
    core::{make_shared, run, Evaluation, SerialExperiment},
    domains::{ContinuousMountainCar, Domain},
    fa::{basis::fixed::Fourier, LFA},
    logging,
    policies::parameterised::{Dirac, Gaussian1d},
    prediction::td::TD,
};

fn main() {
    let domain = ContinuousMountainCar::default();
    let bases = Fourier::from_space(3, domain.state_space());

    let mean_fa = make_shared(LFA::scalar(bases.clone()));

    // Build a stochastic behaviour policy with exponential epsilon.
    let target_policy = make_shared(Dirac::new(mean_fa.clone()));
    let behaviour_policy = make_shared(Gaussian1d::new(mean_fa, 1.0));
    let critic = make_shared(TD::new(make_shared(LFA::scalar(bases)), 0.01, 1.0));

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
