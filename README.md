# RSRL ([api](https://tspooner.github.io/rsrl))

[![Build Status](https://travis-ci.org/tspooner/rsrl.svg?branch=master)](https://travis-ci.org/tspooner/rsrl) [![Coverage Status](https://coveralls.io/repos/github/tspooner/rsrl/badge.svg?branch=master)](https://coveralls.io/github/tspooner/rsrl?branch=master)

## Summary
The ``rsrl`` crate provides generic constructs for running reinforcement
learning (RL) experiments. The main objective of the project is to provide a
simple, extensible framework to investigate new algorithms and methods for
solving learning problems. It aims to combine _speed_, _safety_ and _ease of
use_.


## Example
The [examples](https://github.com/tspooner/rsrl/tree/master/examples) directory
will be helpful here.

See below for a simple example script which makes use of the
[GreedyGQ](http://old.sztaki.hu/~szcsaba/papers/ICML10_controlGQ.pdf) algorithm
with using a fourier basis function approximator to represent the action-value
function applied to the canonical mountain car problem.

```Rust
extern crate rsrl;
#[macro_use] extern crate slog;

use rsrl::{run, logging, Parameter, SerialExperiment, Evaluation};
use rsrl::agents::control::gtd::GreedyGQ;
use rsrl::domains::{Domain, MountainCar};
use rsrl::fa::Linear;
use rsrl::fa::projection::Fourier;
use rsrl::geometry::Space;
use rsrl::policies::EpsilonGreedy;


fn main() {
    let logger = logging::root(logging::stdout());

    let domain = MountainCar::default();
    let mut agent = {
        let n_actions = domain.action_space().span().into();

        // Build the linear value functions using a fourier basis projection.
        let v_func = Linear::new(Fourier::from_space(3, domain.state_space()), 1);
        let q_func = Linear::new(Fourier::from_space(3, domain.state_space()), n_actions);

        // Build a stochastic behaviour policy with exponentially decaying epsilon.
        let policy = EpsilonGreedy::new(Parameter::exponential(0.99, 0.05, 0.99));

        GreedyGQ::new(q_func, v_func, policy, 1e-1, 1e-3, 0.99)
    };

    // Training phase:
    let _training_result = {
        // Build a serial learning experiment with a maximum of 1000 steps per episode.
        let e = SerialExperiment::new(&mut agent, Box::new(MountainCar::default), 1000);

        // Realise 5000 episodes of the experiment generator.
        run(e, 1000, Some(logger.clone()))
    };

    // Testing phase:
    let testing_result =
        Evaluation::new(&mut agent, Box::new(MountainCar::default)).next().unwrap();

    info!(logger, "solution"; testing_result);
}
```
