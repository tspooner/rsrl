# RSRL

[![Build Status](https://travis-ci.org/tspooner/rsrl.svg?branch=master)](https://travis-ci.org/tspooner/rsrl) [![Coverage Status](https://coveralls.io/repos/github/tspooner/rsrl/badge.svg?branch=master)](https://coveralls.io/github/tspooner/rsrl?branch=master) [![Issue Count](https://codeclimate.com/github/tspooner/rsrl/badges/gpa.svg)](https://codeclimate.com/github/tspooner/rsrl)

## Summary

The ``rsrl`` crate provides generic constructs for running reinforcement learning (RL) experiments. It is almost entirely implemented in rust but supports the use of external C/C++ code via an FFI.

The main objective of the project is to provide a simple, extensible framework to investigate new algorithms and methods for solving learning problems. It aims to combine _speed_, _safety_ and _ease of use_.

## Status

The crate is still very young, so the API is likely to change in the future. So far the only algorithms implemented are relatively basic - the intention is to extend deeply into deep-learning techniques and the state-of-the-art linear methods.

We have a way to go yet... :wink:

## Usage

The crate is still very much in alpha so we leave it up to you to clone if you like, but it isn't yet available on [crates.io](https://crates.io/). For example, one can add it to your ``Cargo.toml``,

```
[dependencies]
rsrl = {git = "https://github.com/tspooner/rsrl.git"}
```

and then load it in to your project,
```Rust
extern crate rsrl;
```


### Example
The [examples](https://github.com/tspooner/rsrl/tree/master/examples) directory will be helpful here.

See below for a simple example script which uses the [GreedyGQ](http://old.sztaki.hu/~szcsaba/papers/ICML10_controlGQ.pdf) algorithm with radial basis function networks function approximators to solve the canonical mountain car problem.

```Rust
extern crate rsrl;

use rsrl::{run, SerialExperiment, Evaluation};
use rsrl::fa::linear::RBFNetwork;
use rsrl::domain::{Domain, MountainCar};
use rsrl::agents::td::GreedyGQ;
use rsrl::policies::{Greedy, EpsilonGreedy};
use rsrl::geometry::Space;

use rsrl::logging::DefaultLogger;


fn main() {
    let domain = MountainCar::default();
    let mut agent = {
        let aspace = domain.action_space();
        let n_actions: usize = aspace.span().into();

        let q_func = RBFNetwork::new(
            domain.state_space().partitioned(8), n_actions);
        let v_func = RBFNetwork::new(
            domain.state_space().partitioned(8), 1);

        GreedyGQ::new(q_func, v_func, Greedy, 0.99, 0.1, 1e-5)
    };

    // Training:
    let training_result = {
        let e = SerialExperiment::new(&mut agent,
                                      Box::new(MountainCar::default),
                                      1000);

        run(e, 1000)
    };

    // Testing:
    let testing_result =
        Evaluation::new(&mut agent, Box::new(MountainCar::default)).next().unwrap();


    println!("Solution \u{21D2} {} steps | reward {}",
             testing_result.n_steps,
             testing_result.total_reward);
}
```
