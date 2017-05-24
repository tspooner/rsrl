extern crate rsrl;

use rsrl::{run, SerialExperiment, Evaluation};
use rsrl::fa::linear::RBFNetwork;
use rsrl::agents::control::td::QLearning;
use rsrl::domains::{Domain, MountainCar};
use rsrl::policies::{Greedy, EpsilonGreedy};
use rsrl::geometry::Space;

use rsrl::logging::DefaultLogger;


fn main() {
    let domain = MountainCar::default();
    let mut agent = {
        let aspace = domain.action_space();
        let n_actions: usize = aspace.span().into();

        let q_func = RBFNetwork::new(
            domain.state_space().partitioned(8), n_actions);

        QLearning::new(q_func, Greedy, 0.1, 0.95)
    };

    // Training:
    let training_result = {
        let e = SerialExperiment::new(&mut agent,
                                      Box::new(MountainCar::default),
                                      1000);

        run(e, 1000)
    };

    // Testing:
    let testing_result =
        Evaluation::new(&mut agent, Box::new(MountainCar::default)).next().unwrap();


    println!("Solution \u{21D2} {} steps | reward {}",
             testing_result.n_steps,
             testing_result.total_reward);
}
