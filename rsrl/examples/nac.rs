extern crate rsrl;

use rand::thread_rng;
use rsrl::{
    control::{ac::NAC, td::SARSA},
    domains::{ContinuousMountainCar, Domain},
    fa::{
        linear::{
            basis::{Combinators, Fourier, SCB},
            optim::SGD,
            LFA,
        },
        transforms::Softplus,
        Composition,
    },
    make_shared,
    policies::{Gaussian, Policy},
    Handler,
};
use spaces::BoundedSpace;

fn main() {
    let domain = ContinuousMountainCar::default();

    let limits = domain
        .state_space()
        .into_iter()
        .map(|d| (d.inf().unwrap(), d.sup().unwrap()))
        .collect();

    let basis = Fourier::new(3, limits).with_bias();
    let lfa = LFA::scalar(basis.clone(), SGD(1.0));
    let clfa = Composition::new(LFA::scalar(basis.clone(), SGD(1.0)), Softplus);

    let policy = make_shared(Gaussian::new(lfa, clfa));
    let critic = {
        let optimiser = SGD(0.001);

        let basis_c = SCB {
            policy: policy.clone(),
            basis,
        };
        let cfa = LFA::scalar(basis_c, optimiser);

        SARSA::new(cfa, policy.clone(), 0.999)
    };

    let mut rng = thread_rng();
    let mut agent = NAC::new(critic, policy, 0.001);

    for e in 0..1000 {
        // Episode loop:
        let mut env = ContinuousMountainCar::default();
        let mut action = agent.policy.sample(&mut rng, env.emit().state());
        let mut total_reward = 0.0;

        for i in 0..1000 {
            // Trajectory loop:
            let t = env.transition(action);

            agent.critic.handle(&t).ok();
            action = agent.policy.sample(&mut rng, t.to.state());
            total_reward += t.reward;

            if i % 100 == 0 {
                agent.handle(()).ok();
            }

            if t.terminated() {
                break;
            }
        }

        println!("Batch {}: {}", e + 1, total_reward);
    }

    let traj =
        ContinuousMountainCar::default().rollout(|s| agent.policy.mode(s), Some(1000));

    println!("OOS: {}...", traj.total_reward());
}
