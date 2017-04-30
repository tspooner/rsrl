extern crate rsrl;

use rsrl::fa::linear::RBFNetwork;
use rsrl::domain::{Domain, MountainCar};
use rsrl::agents::td::QLearning;
use rsrl::policies::{Greedy, EpsilonGreedy};
use rsrl::geometry::Space;
use rsrl::experiment::{run, SerialExperiment, Evaluation};

use rsrl::loggers::DefaultLogger;


fn main() {
    let _ = DefaultLogger::init();

    let domain = MountainCar::default();
    let mut agent = {
        let aspace = domain.action_space();
        let n_actions: usize = aspace.span().into();

        let q_func = RBFNetwork::new(
            domain.state_space().with_partitions(8), n_actions);

        QLearning::new(q_func, Greedy, 0.1, 0.99)
    };

    // Training:
    let training_result = {
        let e = SerialExperiment::new(&mut agent,
                                      Box::new(MountainCar::default),
                                      1000);

        run(e, 1000)
    };

    // Testing:
    let testing_result = {
        let e = Evaluation::new(&mut agent, Box::new(MountainCar::default));

        run(e, 1)
    };
}
