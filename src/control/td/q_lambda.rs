use core::{Algorithm, Controller, Predictor, Shared, Parameter, Vector, Matrix, Trace};
use domains::Transition;
use fa::{Approximator, Parameterised, MultiLFA, Projection, Projector, QFunction};
use policies::{fixed::Greedy, Policy};
use std::marker::PhantomData;

/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLambda<S, M: Projector<S>, P: Policy<S>> {
    trace: Trace,

    pub fa_theta: Shared<MultiLFA<S, M>>,

    pub policy: Shared<P>,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: 'static, M: Projector<S> + 'static, P: Policy<S>> QLambda<S, M, P> {
    pub fn new<T1, T2>(
        trace: Trace,
        fa_theta: Shared<MultiLFA<S, M>>,
        policy: Shared<P>,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        QLambda {
            trace,

            fa_theta: fa_theta.clone(),

            policy,
            target: Greedy::new(fa_theta),

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> QLambda<S, M, P> {
    #[inline(always)]
    fn update_theta(&mut self, action: P::Action, error: f64) {
        self.fa_theta.borrow_mut().update_action_phi(
            &Projection::Dense(self.trace.get()),
            action,
            self.alpha * error,
        );
    }

    #[inline(always)]
    fn decay_trace(&mut self, state: &S, action: P::Action) {
        if action == self.target.sample(state) {
            let rate = self.trace.lambda.value() * self.gamma.value();
            self.trace.decay(rate);
        } else {
            self.trace.decay(0.0);
        }
    }

    #[inline(always)]
    fn update_trace(&mut self, phi_s: Projection) {
        self.trace.update(&phi_s.expanded(self.fa_theta.borrow().projector.dim()));
    }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> Algorithm<S, P::Action> for QLambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.borrow().projector.project(s);

        let na = self.target.sample(&ns);
        let qsa = self.fa_theta.borrow().evaluate_action_phi(&phi_s, t.action);
        let nqsna = self.fa_theta.borrow().evaluate_action(ns, na);

        let td_error = t.reward + self.gamma * nqsna - qsa;

        self.decay_trace(s, t.action);
        self.update_trace(phi_s);
        self.update_theta(t.action, td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, P::Action>) {
        {
            let s = t.from.state();
            let phi_s = self.fa_theta.borrow().projector.project(s);
            let qsa = self.fa_theta.borrow().evaluate_action_phi(&phi_s, t.action);

            self.decay_trace(s, t.action);
            self.update_trace(phi_s);
            self.update_theta(t.action, t.reward - qsa);
        }

        self.target.handle_terminal(t);
        self.policy.borrow_mut().handle_terminal(t);

        self.trace.decay(0.0);

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> Controller<S, P::Action> for QLambda<S, M, P> {
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> Predictor<S, P::Action> for QLambda<S, M, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        let a = self.sample_target(s);

        self.predict_qsa(s, a)
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.fa_theta.borrow().evaluate_action(&s, a)
    }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> Parameterised for QLambda<S, M, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
