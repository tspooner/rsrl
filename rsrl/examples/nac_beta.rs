extern crate rsrl;

use rand::thread_rng;
use rsrl::{
    control::{nac::NAC, td::SARSA},
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
    policies::{Beta, Policy},
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
    let lfa = Composition::new(LFA::scalar(basis.clone(), SGD(1.0)), Softplus);

    let policy = make_shared(Beta::new(lfa.clone(), lfa));
    let critic = {
        let optimiser = SGD(0.01);

        let basis_c = SCB {
            policy: policy.clone(),
            basis,
        };
        let cfa = LFA::scalar(basis_c, optimiser);

        SARSA {
            q_func: cfa,
            policy: policy.clone(),

            gamma: 0.999,
        }
    };

    let mut rng = thread_rng();
    let mut agent = NAC::new(critic, policy, 0.1);

    for e in 0..1000 {
        // Episode loop:
        let mut env = ContinuousMountainCar::default();
        let mut action = agent.policy.sample(&mut rng, env.emit().state());
        let mut total_reward = 0.0;

        for i in 0..1000 {
            // Trajectory loop:
            let t = env.transition(2.0 * action - 1.0).replace_action(action);

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
        ContinuousMountainCar::default().rollout(|s| 2.0 * agent.policy.mode(s) - 1.0, Some(1000));

    println!("OOS: {}...", traj.total_reward());
}
