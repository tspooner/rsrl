extern crate rsrl;

use rsrl::{Function, Parameterised};
use rsrl::fa::RBFNetwork;
use rsrl::domain::{Domain, Observation, MountainCar};
use rsrl::agents::Agent;
use rsrl::agents::td_zero::SARSA;
use rsrl::agents::actor_critic::ActorCritic;
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

    let actor = RBFNetwork::new(
        domain.state_space().with_partitions(10), n_actions);
    let critic = RBFNetwork::new(
        domain.state_space().with_partitions(10), 1);

    let mut agent = ActorCritic::new(actor, critic, Greedy, 0.01, 0.05, 0.95);

    let mut a = match domain.emit() {
        Observation::Full(ref s) => Greedy.sample(s),
        _ => panic!("FooBar"),
    };

    for e in 1..2000 {
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
