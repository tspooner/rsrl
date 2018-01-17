extern crate rsrl;
#[macro_use] extern crate slog;

use rsrl::{run, logging, Parameter, SerialExperiment, Evaluation};
use rsrl::agents::memory::Trace;
use rsrl::agents::prediction::gtd::GTD2;
use rsrl::agents::control::actor_critic::ActorCritic;
use rsrl::domains::{Domain, MountainCar};
use rsrl::fa::{Linear, Projector};
use rsrl::fa::projection::Fourier;
use rsrl::geometry::Space;
use rsrl::policies::EpsilonGreedy;


fn main() {
    let domain = MountainCar::default();
    let mut agent = {
        let n_actions = domain.action_space().span().into();

        // Build the basis projector used by the actor and critic.
        let bases = Fourier::from_space(3, domain.state_space());

        // Build the actor.
        let q_func = Linear::new(bases.clone(), n_actions);

        // Build the critic.
        let v_func = Linear::new(bases.clone(), 1);
        let a_func = Linear::new(bases, 1);
        let critic = GTD2::new(v_func, a_func, 0.01, 0.001, 0.99);

        // Build a stochastic behaviour policy with exponential epsilon.
        let eps = Parameter::exponential(0.99, 0.05, 0.99);
        let policy = EpsilonGreedy::new(eps);

        ActorCritic::new(q_func, critic, policy, 0.001, 0.99)
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
    let testing_result =
        Evaluation::new(&mut agent, domain_builder).next().unwrap();

    info!(logger, "solution"; testing_result);
}
