use core::{Handler, Controller, Predictor, Shared, Parameter, Vector, Trace};
use domains::Transition;
use fa::{Approximator, MultiLFA, Projection, Projector, QFunction};
use policies::{fixed::Greedy, Policy, FinitePolicy};
use std::marker::PhantomData;

/// True online variant of the Q(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
pub struct TOQLambda<S, M: Projector<S>, P: FinitePolicy<S>> {
    trace: Trace,
    q_old: f64,

    pub q_func: Shared<MultiLFA<S, M>>,

    pub policy: P,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: 'static, M, P> TOQLambda<S, M, P>
where
    M: Projector<S> + 'static,
    P: FinitePolicy<S>,
{
    pub fn new<T1, T2>(
        trace: Trace,
        q_func: Shared<MultiLFA<S, M>>,
        policy: P,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TOQLambda {
            trace: trace,
            q_old: 0.0,

            q_func: q_func.clone(),

            policy: policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Handler<Transition<S, usize>> for TOQLambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.borrow().projector.project(s);

        let na = self.target.sample(&ns);
        let qsa = self.q_func.borrow().evaluate_action_phi(&phi_s, t.action);
        let nqa = self.q_func.borrow().evaluate_action(ns, na);

        let td_error = t.reward + self.gamma * nqa - qsa;
        let phi_s = phi_s.expanded(self.q_func.borrow().projector.dim());

        let rate = self.trace.lambda.value() * self.gamma.value();
        let trace_update =
            (1.0 - rate * self.alpha.value() * self.trace.get().dot(&phi_s)) * phi_s.clone();

        if t.action == self.target.sample(&s) {
            let rate = self.trace.lambda.value() * self.gamma.value();
            self.trace.decay(rate);
        } else {
            self.trace.decay(0.0);
        }

        self.trace.update(&trace_update);
        self.q_func.borrow_mut().update_action_phi(
            &Projection::Dense(self.trace.get()),
            t.action,
            self.alpha * (td_error + qsa - self.q_old),
        );
        self.q_func.borrow_mut().update_action_phi(
            &Projection::Dense(phi_s),
            t.action,
            self.alpha * (self.q_old - qsa),
        );

        self.q_old = nqa;
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
        self.policy.handle_terminal(t);
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Controller<S, usize> for TOQLambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize { self.target.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.policy.sample(s) }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Predictor<S, usize> for TOQLambda<S, M, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        let a = self.pi(s);

        self.predict_qsa(s, a)
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: usize) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}
