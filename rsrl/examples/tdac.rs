#[macro_use]
extern crate rsrl;

use rand::thread_rng;
use rsrl::{
    control::ac::ActorCritic,
    domains::{ContinuousMountainCar, Domain},
    fa::linear::{
        basis::{Basis, Combinators, Fourier},
        optim::SGD,
        LFA,
    },
    make_shared,
    policies::{Gaussian, Policy},
    prediction::lstd::iLSTD,
    Handler,
};

fn main() {
    let domain = ContinuousMountainCar::default();
    let basis = Fourier::from_space(3, domain.state_space()).with_bias();

    let lfa = LFA::scalar(basis.clone(), SGD(1.0));

    let policy = Gaussian::new(lfa, 1.0);
    let mut eval = shared!(iLSTD::new(basis, 0.0001, 0.99, 2));

    let critic = {
        let e = eval.clone();

        move |(s,): (&_,)| e.basis.project(s).unwrap().dot(&e.theta)
    };

    let mut rng = thread_rng();
    let mut agent = ActorCritic::tdac(critic, policy, 0.002, 0.99);

    for e in 0..100 {
        // Episode loop:
        let mut env = ContinuousMountainCar::default();
        let mut action = agent.policy.sample(&mut rng, env.emit().state());
        let mut total_reward = 0.0;

        for _ in 0..1000 {
            // Trajectory loop:
            let t = env.transition(action).replace_action(action);

            eval.handle(&t).ok();
            agent.handle(&t).ok();

            action = agent.policy.sample(&mut rng, t.to.state());
            total_reward += t.reward;

            if t.terminated() {
                break;
            }
        }

        println!("Batch {}: {}", e + 1, total_reward);
    }

    let traj = ContinuousMountainCar::default().rollout(|s| agent.policy.mode(s), Some(1000));

    println!("OOS: {}...", traj.total_reward());
}
