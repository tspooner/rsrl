extern crate rsrl;
#[macro_use] extern crate slog;

use rsrl::{run, logging, Parameter, SerialExperiment, Evaluation};
use rsrl::agents::memory::Trace;
use rsrl::agents::control::lspi::LSPILambda;
use rsrl::domains::{Domain, MountainCar};
use rsrl::fa::{Linear, Projector};
use rsrl::fa::projection::Fourier;
use rsrl::geometry::Space;
use rsrl::policies::EpsilonGreedy;


fn main() {
    let domain = MountainCar::default();
    let mut agent = {
        let n_actions = domain.action_space().span().into();

        let bases = Fourier::from_space(3, domain.state_space());
        let trace = Trace::replacing(0.7, bases.activation());
        let q_func = Linear::new(bases, n_actions);

        let eps = Parameter::exponential(0.99, 0.05, 0.99);
        let policy = EpsilonGreedy::new(eps);

        LSPILambda::new(trace, q_func, policy, 0.99, 0.70)
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
