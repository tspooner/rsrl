extern crate rsrl;
#[macro_use] extern crate slog;

use rsrl::{run, logging, Parameter, SerialExperiment, Evaluation};
use rsrl::agents::control::gtd::GreedyGQ;
use rsrl::domains::{Domain, OpenAIGym};
use rsrl::fa::Linear;
use rsrl::fa::projection::Fourier;
use rsrl::geometry::Space;
use rsrl::geometry::dimensions::BoundedDimension;
use rsrl::policies::EpsilonGreedy;


fn main() {
    let logger = logging::root(logging::stdout());

    let domain = OpenAIGym::new("CartPole-v1").unwrap();
    let mut agent = {
        let n_actions = domain.action_space().span().into();

        // Build the linear value functions using a fourier basis projection.
        let v_func = Linear::new(Fourier::from_space(3, domain.state_space()), 1);
        let q_func = Linear::new(Fourier::from_space(3, domain.state_space()), n_actions);

        // Build a stochastic behaviour policy with exponentially decaying epsilon.
        let policy = EpsilonGreedy::new(Parameter::exponential(0.99, 0.05, 0.99));

        GreedyGQ::new(q_func, v_func, policy, 1e-1, 1e-3, 0.99)
    };

    // Training phase:
    let _training_result = {
        // Build a serial learning experiment with a maximum of 1000 steps per episode.
        let e = SerialExperiment::new(&mut agent,
                                      Box::new(|| OpenAIGym::new("CartPole-v1").unwrap()),
                                      1000);

        // Realise 5000 episodes of the experiment generator.
        run(e, 1000, Some(logger.clone()))
    };

    // Testing phase:
    let testing_result =
        Evaluation::new(&mut agent,
                        Box::new(|| OpenAIGym::new("CartPole-v1").unwrap())).next().unwrap();

    info!(logger, "solution"; testing_result);
}
