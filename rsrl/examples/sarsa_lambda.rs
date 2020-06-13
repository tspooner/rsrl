extern crate rsrl;

use rand::{rngs::StdRng, SeedableRng};
use rsrl::{
    control::td::SARSALambda,
    domains::{Domain, MountainCar},
    fa::linear::{
        basis::{Combinators, Fourier},
        optim::SGD,
        LFA,
    },
    make_shared,
    params::Parameterised,
    policies::{EpsilonGreedy, Greedy, Policy, Random},
    spaces::Space,
    traces::Trace,
    Handler,
};

const ALPHA: f64 = 0.01;
const GAMMA: f64 = 0.99;
const LAMBDA: f64 = 0.7;

fn main() {
    let env = MountainCar::default();

    let mut rng = StdRng::seed_from_u64(0);
    let mut agent = {
        let n_actions = env.action_space().card().into();

        let basis = Fourier::from_space(5, env.state_space()).with_bias();
        let q_func = make_shared(LFA::vector(basis, SGD(1.0), n_actions));

        let policy = EpsilonGreedy::new(Greedy::new(q_func.clone()), Random::new(n_actions), 0.2);
        let wdim = q_func.weights_dim();

        let trace = Trace::replacing(wdim, GAMMA, LAMBDA);

        SARSALambda::new(q_func, policy, trace, ALPHA, GAMMA)
    };

    for e in 0..1000 {
        // Episode loop:
        let mut j = 0;
        let mut env = MountainCar::default();
        let mut action = agent.policy.sample(&mut rng, env.emit().state());

        for i in 0.. {
            // Trajectory loop:
            j = i;

            let t = env.transition(action);

            agent.handle(&t).ok();
            action = agent.policy.sample(&mut rng, t.to.state());

            if t.terminated() {
                break;
            }
        }

        agent.policy.epsilon *= 0.995;

        println!("Batch {}: {} steps...", e + 1, j + 1);
    }

    let traj = MountainCar::default().rollout(|s| agent.policy.mode(s), Some(1000));

    println!("OOS: {} states...", traj.n_states());
}
