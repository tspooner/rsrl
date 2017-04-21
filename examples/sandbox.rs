extern crate rsrl;

use rsrl::{Function, Parameterised};
use rsrl::fa::RBFNetwork;
use rsrl::domain::{Domain, Observation, MountainCar};
use rsrl::agents::Agent;
use rsrl::agents::td::QLearning;
use rsrl::policies::{Policy, Greedy, EpsilonGreedy};
use rsrl::geometry::{Space, Span};
use rsrl::experiment::{run, SerialExperiment, Evaluation};

use rsrl::loggers::DefaultLogger;


fn main() {
    DefaultLogger::init();

    let mut domain = MountainCar::default();
    let mut agent = {
        let aspace = domain.action_space();
        let n_actions: usize = aspace.span().into();

        let q_func = RBFNetwork::new(
            domain.state_space().with_partitions(8), n_actions);

        QLearning::new(q_func, Greedy, 0.1, 0.99)
    };

    // Training:
    let training_result = {
        let mut e = SerialExperiment::new(&mut agent,
                                          Box::new(MountainCar::default),
                                          1000);

        run(e, 1000)
    };

    // Testing:
    let testing_result = {
        let mut e = Evaluation::new(&mut agent,
                                    Box::new(MountainCar::default));

        run(e, 1)
    };
}
