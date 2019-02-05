extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::actor_critic::TDAC,
    core::{make_shared, run, Evaluation, SerialExperiment},
    domains::{Domain, ContinuousMountainCar},
    fa::{basis::fixed::Fourier, LFA},
    geometry::Space,
    logging,
    prediction::td::TD,
    policies::parameterised::Gaussian1d,
};

fn main() {
    let domain = ContinuousMountainCar::default();
    let bases = Fourier::from_space(3, domain.state_space());

    let policy = make_shared({
        // Build the linear value function using a fourier basis projection and the
        // appropriate eligibility trace.
        let fa = LFA::scalar_output(bases.clone());

        // Build a stochastic behaviour policy with exponential epsilon.
        Gaussian1d::new(fa, 1.0)
    });
    let critic = make_shared({
        // Build the linear value function using a fourier basis projection and the
        // appropriate eligibility trace.
        let v_func = make_shared(LFA::scalar_output(bases));

        TD::new(v_func, 0.01, 0.99)
    });

    let mut agent = TDAC::new(critic, policy, 0.005, 0.99);

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
