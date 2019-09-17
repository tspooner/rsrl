extern crate rsrl;
extern crate rstat;
#[macro_use]
extern crate slog;

use rsrl::{
    control::{ac::NAC, td::SARSA},
    domains::{ContinuousMountainCar, Domain},
    fa::{
        linear::{
            basis::{Chebyshev, Projector},
            optim::SGD,
            StableCFA,
            LFA,
        },
        transforms::Softplus,
        Parameterised,
        TransformedLFA,
    },
    logging,
    make_shared,
    policies::Beta,
    run,
    spaces::Space,
    Evaluation,
    Parameter,
    SerialExperiment,
};

fn main() {
    let domain = ContinuousMountainCar::default();

    let basis = Chebyshev::from_space(5, domain.state_space()).with_constant();
    let policy = Beta::new(
        TransformedLFA::scalar(basis.clone(), Softplus),
        TransformedLFA::scalar(basis.clone(), Softplus),
    );
    let critic = {
        let optimiser = SGD(1.0);
        let q_func = StableCFA::new(policy.clone(), basis, optimiser);

        SARSA::new(q_func, policy.clone(), 0.01, 1.0)
    };

    let mut agent = NAC::new(critic, policy, 0.2);

    let logger = logging::root(logging::stdout());
    let domain_builder = Box::new(ContinuousMountainCar::default);

    // Training phase:
    let _training_result = {
        // Start a serial learning experiment up to 1000 steps per episode.
        let e = SerialExperiment::new(&mut agent, domain_builder.clone(), 1000);

        // Realise 1000 episodes of the experiment generator.
        run(e, 10000, Some(logger.clone()))
    };

    // Testing phase:
    let testing_result = Evaluation::new(&mut agent, domain_builder).next().unwrap();

    info!(logger, "solution"; testing_result);
}
