extern crate rsrl;

use rand::{rngs::StdRng, SeedableRng};
use rsrl::{
    control::td::QLearning,
    domains::{Domain, MountainCar},
    fa::linear::{
        basis::{Combinators, Fourier},
        optim::SGD,
        LFA,
    },
    make_shared,
    policies::{Greedy, Policy},
    spaces::Space,
    Handler,
};

fn main() {
    let env = MountainCar::default();
    let n_actions = env.action_space().card().into();

    let mut rng = StdRng::seed_from_u64(0);
    let (mut ql, policy) = {
        let basis = Fourier::from_space(5, env.state_space()).with_bias();
        let q_func = make_shared(LFA::vector(basis, SGD(0.001), n_actions));
        let policy = Greedy::new(q_func.clone());

        (QLearning { q_func, gamma: 0.9 }, policy)
    };

    for e in 0..200 {
        // Episode loop:
        let mut j = 0;
        let mut env = MountainCar::default();
        let mut action = policy.sample(&mut rng, env.emit().state());

        for i in 0.. {
            // Trajectory loop:
            j = i;

            let t = env.transition(action);

            ql.handle(&t).ok();
            action = policy.sample(&mut rng, t.to.state());

            if t.terminated() {
                break;
            }
        }

        println!("Batch {}: {} steps...", e + 1, j + 1);
    }

    let traj = MountainCar::default().rollout(|s| policy.mode(s), Some(500));

    println!("OOS: {} states...", traj.n_states());
}
