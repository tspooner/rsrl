extern crate rsrl;

use rsrl::{run, Parameter, SerialExperiment, Evaluation};

use rsrl::fa::Linear;
use rsrl::fa::projection::RBFNetwork;

use rsrl::agents::control::td::QSigma;
use rsrl::domains::{Domain, MountainCar};
use rsrl::policies::{Greedy, EpsilonGreedy};
use rsrl::geometry::Space;

use rsrl::logging;
use std::fs::OpenOptions;


fn main() {
    let log_path = "/tmp/log_example.log";
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(log_path)
        .unwrap();

    let logger = logging::root(
        logging::combine(logging::stdout(), logging::file(file)));

    let domain = MountainCar::default();
    let mut agent = {
        let aspace = domain.action_space();
        let n_actions: usize = aspace.span().into();

        let pr = RBFNetwork::from_space(domain.state_space().partitioned(8));
        let q_func = Linear::new(pr, n_actions);

        let policy = EpsilonGreedy::new(aspace, Parameter::exponential(0.9, 0.01, 0.99));

        QSigma::new(q_func, policy, 0.05, 0.99, 0.2, 2)
    };

    // Training:
    let training_result = {
        let e = SerialExperiment::new(&mut agent,
                                      Box::new(MountainCar::default),
                                      1000);

        run(e, 1500, Some(logger))
    };

    // Testing:
    let testing_result =
        Evaluation::new(&mut agent, Box::new(MountainCar::default)).next().unwrap();


    println!("Solution \u{21D2} {} steps | reward {}",
             testing_result.steps,
             testing_result.reward);
}
