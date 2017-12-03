extern crate rsrl;
extern crate seahash;

use rsrl::{run, logging, Parameter, SerialExperiment, Evaluation};
use rsrl::agents::control::td::ExpectedSARSA;
use rsrl::domains::{Domain, MountainCar};
use rsrl::fa::{Linear, SparseLinear};
use rsrl::fa::projection::TileCoding;
use rsrl::geometry::Space;
use rsrl::policies::EpsilonGreedy;

use std::hash::BuildHasherDefault;


type SeahashBuildHasher = BuildHasherDefault<seahash::SeaHasher>;


fn main() {
    let logger = logging::root(logging::stdout());

    let domain = MountainCar::default();
    let mut agent = {
        let aspace = domain.action_space();
        let n_actions: usize = aspace.span().into();

        let pr = TileCoding::new(SeahashBuildHasher::default(), 10, 10000);
        let q_func = SparseLinear::new(pr, n_actions);
        let policy = EpsilonGreedy::new(aspace, Parameter::exponential(0.05, 0.05, 0.99));

        ExpectedSARSA::new(q_func, policy, 0.1, 0.99)
    };

    // Training:
    let _training_result = {
        let e = SerialExperiment::new(&mut agent, Box::new(MountainCar::default), 1000);

        run(e, 1000, Some(logger))
    };

    // println!("{:?}", agent.q_func.weights);

    // Testing:
    let testing_result =
        Evaluation::new(&mut agent, Box::new(MountainCar::default)).next().unwrap();


    println!("Solution \u{21D2} {} steps | reward {}",
             testing_result.steps,
             testing_result.reward);
}
