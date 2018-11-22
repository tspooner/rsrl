extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::actor_critic::A2C,
    control::td::SARSA,
    core::{make_shared, run, Evaluation, Parameter, SerialExperiment, Trace},
    domains::{Domain, MountainCar},
    fa::{basis::fixed::Fourier, LFA},
    geometry::Space,
    logging,
    policies::parameterised::Gibbs,
};

fn main() {
    let domain = MountainCar::default();

    let n_actions = domain.action_space().card().into();
    let bases = Fourier::from_space(3, domain.state_space());

    let policy = {
        // Build the linear value function using a fourier basis projection and the
        // appropriate eligibility trace.
        let fa = LFA::vector_valued(bases.clone(), n_actions);

        // Build a stochastic behaviour policy with exponential epsilon.
        make_shared(Gibbs::new(fa))
    };
    let critic = {
        // Build the linear value function using a fourier basis projection and the
        // appropriate eligibility trace.
        let q_func = make_shared(LFA::vector_valued(bases, n_actions));

        SARSA::new(q_func, policy.clone(), 0.001, 0.99)
    };

    let mut agent = A2C::new(critic, policy, 0.01);

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
