extern crate rsrl;

use rsrl::{Function, Parameterised};
use rsrl::fa::RBFNetwork;
use rsrl::domain::{Domain, Observation, MountainCar};
use rsrl::agents::{Agent, QLearning, SARSA};
use rsrl::policies::{Policy, Greedy};
use rsrl::geometry::{Space, Span};

use rsrl::loggers::DefaultLogger;


fn main() {
    DefaultLogger::init();

    let mut domain = MountainCar::default();
    let aspace = domain.action_space();
    let n_actions = match aspace.span() {
        Span::Finite(na) => na,
        _ => panic!("Non-finite action space!")
    };

    let fa = RBFNetwork::new(
        domain.state_space().with_partitions(10), n_actions);

    let mut pi = Greedy;
    let mut a = match domain.emit() {
        Observation::Full(ref s) => pi.sample(s),
        _ => panic!("FooBar"),
    };

    let mut agent = SARSA::new(fa, pi);

    for e in 1..1000 {
        let mut n_steps = 0;
        for j in 1..1000 {
            let t = domain.step(a);

            n_steps = j+1;

            a = match t.to {
                Observation::Terminal(_) => break,
                _ => agent.handle(&t)
            };
        }

        println!("Episode {} - {} Steps", e, n_steps);

        domain = MountainCar::default();
    }
}
