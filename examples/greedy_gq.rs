extern crate rsrl;
#[macro_use] extern crate slog;

use rsrl::{run, logging, Parameter, SerialExperiment, Evaluation};
use rsrl::agents::control::gtd::GreedyGQ;
use rsrl::domains::{Domain, MountainCar};
use rsrl::fa::Linear;
use rsrl::fa::projection::Fourier;
use rsrl::geometry::Space;
use rsrl::policies::EpsilonGreedy;


fn main() {
    let logger = logging::root(logging::stdout());

    let domain = MountainCar::default();
    let mut agent = {
        let n_actions = domain.action_space().span().into();

        // Build the linear value functions using a fourier basis projection.
        let bases = Fourier::from_space(3, domain.state_space());
        let v_func = Linear::new(bases.clone(), 1);
        let q_func = Linear::new(bases, n_actions);

        // Build a stochastic behaviour policy with exponential epsilon.
        let eps = Parameter::exponential(0.99, 0.05, 0.99);
        let policy = EpsilonGreedy::new(eps);

        GreedyGQ::new(q_func, v_func, policy, 1e-1, 1e-3, 0.99)
    };

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
