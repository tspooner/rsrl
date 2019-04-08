extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::{actor_critic::A2C, td::SARSA},
    core::{make_shared, run, Evaluation, SerialExperiment},
    domains::{Domain, MountainCar},
    fa::{basis::{Composable, fixed::Fourier}, LFA},
    geometry::Space,
    logging,
    policies::parameterised::Gibbs,
};

fn main() {
    let domain = MountainCar::default();

    let n_actions = domain.action_space().card().into();
    let bases = Fourier::from_space(3, domain.state_space()).with_constant();

    let policy = make_shared({
        let fa = LFA::vector(bases.clone(), n_actions);

        Gibbs::new(fa)
    });
    let critic = {
        let q_func = LFA::vector(bases, n_actions);

        SARSA::new(q_func, policy.clone(), 0.1, 0.99)
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
