extern crate rsrl;
extern crate fxhash;

use rsrl::{run, Parameter, SerialExperiment, Evaluation};

use rsrl::agents::control::td::ExpectedSARSA;
use rsrl::domains::{Domain, MountainCar};

use rsrl::fa::{Linear, SparseLinear};
use rsrl::fa::projection::TileCoding;
use rsrl::geometry::Space;

use rsrl::logging;
use rsrl::policies::EpsilonGreedy;
use std::fs::OpenOptions;


fn main() {
    // let log_path = "/tmp/log_example.log";
    // let file = OpenOptions::new()
        // .create(true)
        // .write(true)
        // .truncate(true)
        // .open(log_path)
        // .unwrap();

    // let logger = logging::root(logging::combine(logging::stdout(), logging::file(file)));

    let logger = logging::root(logging::stdout());

    let domain = MountainCar::default();
    let mut agent = {
        let aspace = domain.action_space();
        let n_actions: usize = aspace.span().into();

        let pr = TileCoding::new(fxhash::FxBuildHasher::default(), 8, 100000);
        let q_func = SparseLinear::new(pr, n_actions);
        let policy = EpsilonGreedy::new(aspace, Parameter::exponential(0.9, 0.01, 0.99));

        ExpectedSARSA::new(q_func, policy, 0.01, 0.95)
    };

    // Training:
    let _training_result = {
        let e = SerialExperiment::new(&mut agent, Box::new(MountainCar::default), 1000);

        run(e, 1500, Some(logger))
    };

    // Testing:
    let testing_result =
        Evaluation::new(&mut agent, Box::new(MountainCar::default)).next().unwrap();


    println!("Solution \u{21D2} {} steps | reward {}",
             testing_result.steps,
             testing_result.reward);
}
