extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    logging, run, Evaluation, Parameter, SerialExperiment,
    agents::control::td::PAL,
    domains::{Domain, MountainCar},
    fa::{MultiLinear, projection::Fourier},
    geometry::Space,
    policies::EpsilonGreedy
};

fn main() {
    let domain = MountainCar::default();
    let mut agent = {
        let n_actions = domain.action_space().span().into();

        // Build the linear value function using a fourier basis projection and the
        // appropriate eligibility trace.
        let bases = Fourier::from_space(3, domain.state_space());
        let q_func = MultiLinear::new(bases, n_actions);

        // Build a stochastic behaviour policy with exponential epsilon.
        let eps = Parameter::exponential(0.99, 0.05, 0.99);
        let policy = EpsilonGreedy::new(eps);

        PAL::new(q_func, policy, 0.1, 0.99)
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
