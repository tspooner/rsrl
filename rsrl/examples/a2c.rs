#[macro_use]
extern crate rsrl;

use rand::thread_rng;
use rsrl::{
    control::{ac::ActorCritic, td::SARSA},
    domains::{Domain, MountainCar, Transition},
    fa::linear::{
        basis::{Combinators, Fourier},
        optim::SGD,
        LFA,
    },
    make_shared,
    policies::{Gibbs, Policy},
    spaces::Space,
    Function,
    Handler,
};

fn main() {
    let domain = MountainCar::default();
    let n_actions = domain.action_space().card().into();

    let basis = Fourier::from_space(3, domain.state_space()).with_bias();

    let q_func = shared!(LFA::vector(basis.clone(), SGD(0.001), n_actions));
    let policy = shared!({
        let fa = LFA::vector(basis, SGD(1.0), n_actions);

        Gibbs::standard(fa)
    });

    let mut eval = SARSA {
        q_func: q_func.clone(),
        policy: policy.clone(),
        gamma: 1.0,
    };
    let critic = {
        let q = q_func.clone();
        let p = policy.clone();

        move |t: &Transition<_, _>| {
            let qs = q.evaluate((t.from.state(),));
            let ps = p.evaluate((t.from.state(),));

            qs[t.action] - qs.into_iter().zip(ps.into_iter()).fold(0.0, |a, (x, p)| a + x * p)
        }
    };

    let mut rng = thread_rng();
    let mut agent = ActorCritic::new(critic, policy, 0.001);

    for e in 0..1000 {
        // Episode loop:
        let mut env = MountainCar::default();
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

    let traj = MountainCar::default().rollout(|s| agent.policy.mode(s), Some(1000));

    println!("OOS: {}...", traj.total_reward());
}
