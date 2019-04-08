extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::gtd::GreedyGQ,
    core::{make_shared, run, Evaluation, Parameter, SerialExperiment},
    domains::{Domain, MountainCar},
    fa::{basis::{Composable, fixed::Fourier}, LFA},
    geometry::Space,
    logging,
    policies::fixed::{EpsilonGreedy, Greedy, Random},
};

fn main() {
    let logger = logging::root(logging::stdout());

    let domain = MountainCar::default();
    let mut agent = {
        let n_actions = domain.action_space().card().into();

        // Build the linear value functions using a fourier basis projection.
        let bases = Fourier::from_space(3, domain.state_space()).with_constant();
        let v_func = LFA::scalar(bases.clone());
        let q_func = make_shared(LFA::vector(bases, n_actions));

        // Build a stochastic behaviour policy with exponential epsilon.
        let policy = EpsilonGreedy::new(
            Greedy::new(q_func.clone()),
            Random::new(n_actions),
            Parameter::exponential(0.2, 0.0001, 0.99),
        );

        GreedyGQ::new(q_func, v_func, policy, 0.01, 0.001, 0.99)
    };

    let domain_builder = Box::new(MountainCar::default);

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
