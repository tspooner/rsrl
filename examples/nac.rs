extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    make_shared, run, Evaluation, SerialExperiment, Parameter,
    control::{ac::NAC, td::SARSA},
    domains::{Domain, ContinuousMountainCar},
    fa::{
        Parameterised,
        linear::{LFA, StableCFA, basis::{Projector, Fourier}, optim::SGD},
    },
    logging,
    policies::gaussian::{self, Gaussian},
    spaces::Space,
};

fn main() {
    let domain = ContinuousMountainCar::default();

    let basis = Fourier::from_space(3, domain.state_space()).with_constant();
    let policy = make_shared(Gaussian::new(
        gaussian::mean::Scalar(LFA::scalar(basis.clone(), SGD(1.0))),
        gaussian::stddev::Constant(0.5),
    ));
    let critic = {
        let q_func = StableCFA::new(policy.clone(), basis, SGD(1.0));

        SARSA::new(q_func, policy.clone(), 0.01, 1.0)
    };

    let mut agent = NAC::new(critic, policy, 0.01);

    let logger = logging::root(logging::stdout());
    let domain_builder = Box::new(ContinuousMountainCar::default);

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
