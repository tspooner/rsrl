extern crate rsrl;

use rand::{rngs::StdRng, SeedableRng};
use rsrl::{
    control::td::GreedyGQ,
    domains::{Domain, MountainCar},
    fa::linear::{
        basis::{Combinators, Fourier},
        optim::SGD,
        LFA,
    },
    make_shared,
    policies::{EpsilonGreedy, Greedy, Policy, Random},
    spaces::Space,
    Enumerable,
    Handler,
};

fn main() {
    let env = MountainCar::default();
    let n_actions = env.action_space().card().into();

    let mut rng = StdRng::seed_from_u64(0);
    let mut agent = {
        let basis = Fourier::from_space(3, env.state_space()).with_bias();
        let fa_q = make_shared(LFA::vector(basis.clone(), SGD(0.1), n_actions));
        let fa_td = LFA::vector(basis, SGD(0.001), n_actions);

        let policy = EpsilonGreedy::new(Greedy::new(fa_q.clone()), Random::new(n_actions), 0.1);

        GreedyGQ {
            fa_q,
            fa_td,

            behaviour_policy: policy,
            gamma: 0.99,
        }
    };

    for e in 0..200 {
        // Episode loop:
        let mut j = 0;
        let mut env = MountainCar::default();
        let mut action = agent.behaviour_policy.sample(&mut rng, env.emit().state());

        for i in 0..1000 {
            // Trajectory loop:
            j = i;

            let t = env.transition(action);

            agent.handle(&t).ok();
            action = agent.behaviour_policy.sample(&mut rng, t.to.state());

            if t.terminated() {
                break;
            }
        }

        println!("Batch {}: {} steps...", e + 1, j + 1);
    }

    let traj = MountainCar::default().rollout(|s| agent.fa_q.find_max((s,)).0, Some(500));

    println!("OOS: {} states...", traj.n_states());
}
