extern crate rsrl;

use rand::thread_rng;
use rsrl::{
    control::{ac::A2C, td::SARSA},
    domains::{Domain, MountainCar},
    fa::linear::{
        basis::{Combinators, Fourier},
        optim::SGD,
        LFA,
    },
    make_shared,
    policies::{Gibbs, Policy},
    spaces::Space,
    Handler,
};

fn main() {
    let domain = MountainCar::default();
    let n_actions = domain.action_space().card().into();

    let bases = Fourier::from_space(3, domain.state_space()).with_bias();

    let policy = make_shared({
        let fa = LFA::vector(bases.clone(), SGD(1.0), n_actions);

        Gibbs::standard(fa)
    });
    let critic = {
        let q_func = LFA::vector(bases, SGD(0.001), n_actions);

        SARSA::new(q_func, policy.clone(), 1.0)
    };

    let mut rng = thread_rng();
    let mut agent = A2C::new(critic, policy, 0.001);

    for e in 0..1000 {
        // Episode loop:
        let mut env = MountainCar::default();
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

    let traj = MountainCar::default().rollout(|s| agent.policy.mode(s), Some(1000));

    println!("OOS: {}...", traj.total_reward());
}
