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
rsrl = "0.6"
```

## Usage
The code below shows how one could use `rsrl` to evaluate a QLearning agent
using a linear function approximator with Fourier basis projection to solve the
canonical mountain car problem.

> See [examples/](https://github.com/tspooner/rsrl/tree/master/examples) for
> more...

```rust
extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{
    control::td::QLearning,
    core::{make_shared, run, Evaluation, Parameter, SerialExperiment},
    domains::{Domain, MountainCar},
    fa::linear::{LFA, basis::{Projector, Fourier}, optim::SGD},
    geometry::Space,
    logging,
    policies::{EpsilonGreedy, Greedy, Random},
};

fn main() {
    let domain = MountainCar::default();
    let mut agent = {
        let n_actions = domain.action_space().card().into();

        let basis = Fourier::from_space(5, domain.state_space()).with_constant();
        let optimiser = SGD(1.0);
        let q_func = make_shared(LFA::vector(basis, optimiser, n_actions));

        let policy = EpsilonGreedy::new(
            Greedy::new(q_func.clone()),
            Random::new(n_actions),
            Parameter::exponential(0.5, 0.0, 0.99),
        );

        QLearning::new(q_func, policy, 0.01, 1.0)
    };

    let logger = logging::root(logging::stdout());
    let domain_builder = Box::new(MountainCar::default);

    // Training phase:
    let _training_result = {
        // Start a serial learning experiment up to 1000 steps per episode.
        let e = SerialExperiment::new(&mut agent, domain_builder.clone(), 1000);

        // Realise 1000 episodes of the experiment generator.
        run(e, 1000, Some(logger.clone()))
    };

    // Testing phase:
    let testing_result = Evaluation::new(&mut agent, domain_builder).next().unwrap();

    info!(logger, "solution"; testing_result);
}
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update tests as appropriate and adhere to the angularjs
commit message conventions (see
[here](https://gist.github.com/stephenparish/9941e89d80e2bc58a153)).

## License
[MIT](https://choosealicense.com/licenses/mit/)
