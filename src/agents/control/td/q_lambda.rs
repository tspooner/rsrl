use agents::{Controller, Predictor, memory::Trace};
use domains::Transition;
use fa::{Approximator, MultiLFA, Projection, Projector, QFunction};
use policies::{fixed::Greedy, Policy, FinitePolicy};
use std::marker::PhantomData;
use {Handler, Shared, Parameter};

/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLambda<S, M: Projector<S>, P: FinitePolicy<S>> {
    trace: Trace,

    pub fa_theta: Shared<MultiLFA<S, M>>,

    pub policy: P,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: 'static, M: Projector<S> + 'static, P: FinitePolicy<S>> QLambda<S, M, P> {
    pub fn new<T1, T2>(
        trace: Trace,
        fa_theta: Shared<MultiLFA<S, M>>,
        policy: P,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        QLambda {
            trace: trace,

            fa_theta: fa_theta.clone(),

            policy: policy,
            target: Greedy::new(fa_theta),

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Handler<Transition<S, usize>> for QLambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.borrow().projector.project(s);

        let qs = self.fa_theta.borrow().evaluate_phi(&phi_s);
        let nqs = self.fa_theta.borrow().evaluate(ns).unwrap();

        let td_error = t.reward + self.gamma * nqs[self.target.sample(&ns)] - qs[t.action];

        if t.action == self.target.sample(&s) {
            let rate = self.trace.lambda.value() * self.gamma.value();
            self.trace.decay(rate);
        } else {
            self.trace.decay(0.0);
        }

        self.trace
            .update(&phi_s.expanded(self.fa_theta.borrow().projector.dim()));
        self.fa_theta.borrow_mut().update_action_phi(
            &Projection::Dense(self.trace.get()),
            t.action,
            td_error * self.alpha,
        );
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);

        self.policy.handle_terminal(t);
        self.target.handle_terminal(t);
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Controller<S, usize> for QLambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize { self.policy.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.target.sample(s) }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Predictor<S> for QLambda<S, M, P> {
    fn predict(&mut self, s: &S) -> f64 {
        self.fa_theta.borrow().evaluate(s).unwrap()[self.target.sample(s)]
    }
}
