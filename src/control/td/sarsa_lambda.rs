use core::{Algorithm, Predictor, Controller, Shared, Parameter, Vector, Matrix, Trace};
use domains::Transition;
use fa::{Approximator, Parameterised, MultiLFA, Projection, Projector, QFunction};
use policies::{Policy, FinitePolicy};

/// On-policy variant of Watkins' Q-learning with eligibility traces (aka
/// "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
pub struct SARSALambda<S, M: Projector<S>, P: Policy<S>> {
    pub fa_theta: Shared<MultiLFA<S, M>>,
    pub policy: Shared<P>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    trace: Trace,
}

impl<S, M: Projector<S>, P: Policy<S>> SARSALambda<S, M, P> {
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
        SARSALambda {
            fa_theta,
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            trace,
        }
    }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> SARSALambda<S, M, P> {
    #[inline(always)]
    fn update_theta(&mut self, action: P::Action, error: f64) {
        self.fa_theta.borrow_mut().update_action_phi(
            &Projection::Dense(self.trace.get()),
            action,
            self.alpha * error,
        );
    }

    #[inline(always)]
    fn update_trace(&mut self, phi_s: Projection) {
        let rate = self.trace.lambda.value() * self.gamma.value();

        self.trace.decay(rate);
        self.trace.update(&phi_s.expanded(self.fa_theta.borrow().projector.dim()));
    }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> Algorithm<S, P::Action> for SARSALambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.borrow().projector.project(s);

        let na = self.policy.borrow_mut().sample(ns);
        let qsa = self.fa_theta.borrow().evaluate_action_phi(&phi_s, t.action);
        let nqsna = self.fa_theta.borrow().evaluate_action(ns, na);

        let td_error = t.reward + self.gamma * nqsna - qsa;

        self.update_trace(phi_s);
        self.update_theta(t.action, td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, P::Action>) {
        {
            let s = t.from.state();
            let phi_s = self.fa_theta.borrow().projector.project(s);
            let qsa = self.fa_theta.borrow().evaluate_action_phi(&phi_s, t.action);

            self.update_trace(phi_s);
            self.update_theta(t.action, t.reward - qsa);

            self.trace.decay(0.0);
            self.policy.borrow_mut().handle_terminal(t);
        }

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Controller<S, P::Action> for SARSALambda<S, M, P> {
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Predictor<S, P::Action> for SARSALambda<S, M, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.policy.borrow_mut().probabilities(s))
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.fa_theta.borrow().evaluate_action(&s, a)
    }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> Parameterised for SARSALambda<S, M, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
