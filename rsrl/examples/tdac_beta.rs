extern crate rsrl;

use rand::thread_rng;
use rsrl::{
    control::{ac::TDAC, td::SARSA},
    domains::{ContinuousMountainCar, Domain},
    fa::{
        linear::{
            basis::{Combinators, Fourier},
            optim::SGD,
            LFA,
        },
        transforms::Softplus,
        Composition,
    },
    policies::{Beta, Policy},
    prediction::lstd::iLSTD,
    make_shared,
    spaces::Space,
    Handler,
};

fn main() {
    let domain = ContinuousMountainCar::default();
    let basis = Fourier::from_space(3, domain.state_space()).with_bias();

    let lfa = Composition::new(LFA::scalar(basis.clone(), SGD(1.0)), Softplus);

    let critic = iLSTD::new(basis, 0.00001, 0.999, 2);
    let policy = Beta::new(lfa.clone(), lfa);

    let mut rng = thread_rng();
    let mut agent = TDAC::new(critic, policy, 0.001, 0.999);

    for e in 0..1000 {
        // Episode loop:
        let mut env = ContinuousMountainCar::default();
        let mut action = agent.policy.sample(&mut rng, env.emit().state());
        let mut total_reward = 0.0;

        for _ in 0..1000 {
            // Trajectory loop:
            let t = env.transition(action).replace_action(action);

            agent.critic.handle(&t).ok();
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
