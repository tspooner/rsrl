extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::td::SARSALambda,
    core::{make_shared, run, Evaluation, Parameter, SerialExperiment, Trace},
    domains::{Domain, MountainCar},
    fa::{basis::fixed::Fourier, LFA},
    geometry::Space,
    logging,
    policies::fixed::{EpsilonGreedy, Greedy, Random},
};

fn main() {
    let domain = MountainCar::default();
    let mut agent = {
        let n_actions = domain.action_space().card().into();

        // Build the linear value function using a fourier basis projection and the
        // appropriate eligibility trace.
        let bases = Fourier::from_space(3, domain.state_space());
        let trace = Trace::replacing(0.5, bases.dim());
        let q_func = make_shared(LFA::vector(bases, n_actions));

        // Build a stochastic behaviour policy with exponential epsilon.
        let policy = make_shared(EpsilonGreedy::new(
            Greedy::new(q_func.clone()),
            Random::new(n_actions),
            Parameter::exponential(0.7, 0.001, 0.99),
        ));

        SARSALambda::new(q_func, policy, trace, 0.005, 1.0)
    };

    let logger = logging::root(logging::stdout());
    let domain_builder = Box::new(MountainCar::default);

    // Training phase:
    let _training_result = {
        // Start a serial learning experiment up to 1000 steps per episode.
        let e = SerialExperiment::new(&mut agent, domain_builder.clone(), 1000);

        // Realise 1000 episodes of the experiment generator.
        run(e, 1000, Some(logger.clone()))
    };

    // Testing phase:
    let testing_result = Evaluation::new(&mut agent, domain_builder).next().unwrap();

    info!(logger, "solution"; testing_result);
}
