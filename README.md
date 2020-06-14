<img align="left" width="120" title="RSRL logo" src="https://github.com/tspooner/rsrl/raw/master/logo.png">

# RSRL ([api](https://docs.rs/rsrl/))

[![Crates.io](https://img.shields.io/crates/v/rsrl.svg)](https://crates.io/crates/rsrl)
[![Build Status](https://travis-ci.org/tspooner/rsrl.svg?branch=master)](https://travis-ci.org/tspooner/rsrl)
[![Coverage Status](https://coveralls.io/repos/github/tspooner/rsrl/badge.svg?branch=master)](https://coveralls.io/github/tspooner/rsrl?branch=master)

> Reinforcement learning should be _fast_, _safe_ and _easy to use_.

## Overview

``rsrl`` provides generic constructs for reinforcement learning (RL)
experiments in an extensible framework with efficient implementations of
existing methods for rapid prototyping.

## Installation

```toml
[dependencies]
rsrl = "0.8"
```

Note that `rsrl` enables the `blas` feature of its [`ndarray`] dependency, so
if you're building a binary, you additionally need to specify a BLAS backend
compatible with `ndarray`. For example, you can add these dependencies:

[`ndarray`]: https://crates.io/crates/ndarray

```toml
blas-src = { version = "0.2.0", default-features = false, features = ["openblas"] }
openblas-src = { version = "0.6.0", default-features = false, features = ["cblas", "system"] }
```

See `ndarray`'s [README](https://github.com/rust-ndarray/ndarray#how-to-use-with-cargo)
for more information.

## Usage
The code below shows how one could use `rsrl` to evaluate a QLearning agent
using a linear function approximator with Fourier basis projection to solve the
canonical mountain car problem.

> See [examples/](https://github.com/tspooner/rsrl/tree/master/rsrl/examples) for
> more...

```rust
let env = MountainCar::default();
let n_actions = env.action_space().card().into();

let mut rng = StdRng::seed_from_u64(0);
let (mut ql, policy) = {
    let basis = Fourier::from_space(5, env.state_space()).with_bias();
    let q_func = make_shared(LFA::vector(basis, SGD(0.001), n_actions));
    let policy = Greedy::new(q_func.clone());

    (QLearning {
        q_func,
        gamma: 0.9,
    }, policy)
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
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update tests as appropriate and adhere to the angularjs
commit message conventions (see
[here](https://gist.github.com/stephenparish/9941e89d80e2bc58a153)).

## License
[MIT](https://choosealicense.com/licenses/mit/)
