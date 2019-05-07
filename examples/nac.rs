extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::{actor_critic::NAC, td::SARSA},
    core::{make_shared, run, Evaluation, SerialExperiment},
    domains::{Domain, MountainCar},
    fa::{basis::fixed::Fourier, Composable, LFA},
    geometry::Space,
    logging,
    policies::Gibbs,
};

fn main() {
    let domain = MountainCar::default();

    let n_actions = domain.action_space().card().into();
    let bases = Fourier::from_space(3, domain.state_space()).with_constant();

    let policy = make_shared(Gibbs::standard(LFA::vector(bases.clone(), n_actions)));
    let critic = {
        let q_func = LFA::vector(bases, n_actions);

        SARSA::new(q_func, policy.clone(), 0.001, 0.99)
    };

    let mut agent = NAC::new(critic, policy, 0.0001);

    let logger = logging::root(logging::stdout());
    let domain_builder = Box::new(MountainCar::default);

    // Training phase:
    let _training_result = {
        // Start a serial learning experiment up to 1000 steps per episode.
        let e = SerialExperiment::new(&mut agent, domain_builder.clone(), 1000);

        // Realise 1000 episodes of the experiment generator.
        run(e, 500, Some(logger.clone()))
    };

    // Testing phase:
    let testing_result = Evaluation::new(&mut agent, domain_builder).next().unwrap();

    info!(logger, "solution"; testing_result);
}
