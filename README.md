<img align="right" width="120" title="RSRL logo" src="https://github.com/tspooner/rsrl/raw/master/logo.png">

# RSRL ([api](https://tspooner.github.io/rsrl))

[![Crates.io](https://img.shields.io/crates/v/rsrl.svg)](https://crates.io/crates/rsrl)
[![Build Status](https://travis-ci.org/tspooner/rsrl.svg?branch=master)](https://travis-ci.org/tspooner/rsrl)
[![Coverage Status](https://coveralls.io/repos/github/tspooner/rsrl/badge.svg?branch=master)](https://coveralls.io/github/tspooner/rsrl?branch=master)

> Reinforcement learning should be _fast_, _safe_ and _easy to use_.

## Overview

``rsrl`` provides generic constructs for running reinforcement learning (RL)
experiments by providing a simple, extensible framework and efficient
implementations of existing methods for rapid prototyping.

## Installation
```toml
[dependencies]
rsrl = "0.4"
```

## Usage
The code below shows how one could use `rsrl` to evaluate a
[GreedyGQ](http://old.sztaki.hu/~szcsaba/papers/ICML10_controlGQ.pdf) agent
using a Fourier basis function approximator to solve the canonical mountain car
problem.

> See [examples/](https://github.com/tspooner/rsrl/tree/master/examples) for
> more...

```rust
extern crate rsrl;
#[macro_use]
extern crate slog;

use rsrl::{logging, run, Evaluation, Parameter, SerialExperiment};
use rsrl::agents::control::gtd::GreedyGQ;
use rsrl::domains::{Domain, MountainCar};
use rsrl::fa::{MultiLinear, SimpleLinear};
use rsrl::fa::projection::Fourier;
use rsrl::geometry::Space;
use rsrl::policies::EpsilonGreedy;

fn main() {
    let logger = logging::root(logging::stdout());

    let domain = MountainCar::default();
    let mut agent = {
        let n_actions = domain.action_space().span().into();

        // Build the linear value functions using a fourier basis projection.
        let bases = Fourier::from_space(3, domain.state_space());
        let v_func = SimpleLinear::new(bases.clone());
        let q_func = MultiLinear::new(bases, n_actions);

        // Build a stochastic behaviour policy with exponential epsilon.
        let eps = Parameter::exponential(0.99, 0.05, 0.99);
        let policy = EpsilonGreedy::new(eps);

        GreedyGQ::new(q_func, v_func, policy, 1e-1, 1e-3, 0.99)
    };

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

Please make sure to update tests as appropriate and adhere to the angularjs commit message conventions (see [here](https://gist.github.com/stephenparish/9941e89d80e2bc58a153)).

## License
[MIT](https://choosealicense.com/licenses/mit/)
