use core::{Algorithm, Controller, Predictor, Shared, Parameter, Vector, Matrix, Trace};
use domains::Transition;
use fa::{Approximator, Parameterised, MultiLFA, Projection, Projector, QFunction};
use policies::{fixed::Greedy, Policy};
use std::marker::PhantomData;

/// True online variant of the Q(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
pub struct TOQLambda<S, M: Projector<S>, P: Policy<S>> {
    trace: Trace,
    q_old: f64,

    pub q_func: Shared<MultiLFA<S, M>>,

    pub policy: Shared<P>,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: 'static, M, P> TOQLambda<S, M, P>
where
    M: Projector<S> + 'static,
    P: Policy<S>,
{
    pub fn new<T1, T2>(
        trace: Trace,
        q_func: Shared<MultiLFA<S, M>>,
        policy: Shared<P>,
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

impl<S, M, P> Algorithm<S, usize> for TOQLambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = usize>,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.borrow().projector.project(s);

        let na = self.sample_target(&ns);
        let qsa = self.q_func.borrow().evaluate_action_phi(&phi_s, t.action);
        let nqa = self.q_func.borrow().evaluate_action(ns, na);

        let td_error = t.reward + self.gamma * nqa - qsa;
        let phi_s = phi_s.expanded(self.q_func.borrow().projector.dim());

        let rate = self.trace.lambda.value() * self.gamma.value();
        let trace_update =
            (1.0 - rate * self.alpha.value() * self.trace.get().dot(&phi_s)) * phi_s.clone();

        if t.action == self.sample_target(&s) {
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
    }
}

impl<S, M, P> Controller<S, usize> for TOQLambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = usize>,
{
    fn sample_target(&mut self, s: &S) -> usize { self.target.sample(s) }
    fn sample_behaviour(&mut self, s: &S) -> usize { self.policy.borrow_mut().sample(s) }
}

impl<S, M, P> Predictor<S, usize> for TOQLambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = usize>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        let a = self.sample_target(s);

        self.predict_qsa(s, a)
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: usize) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}

impl<S, M, P> Parameterised for TOQLambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = usize>,
{
    fn weights(&self) -> Matrix<f64> {
        self.q_func.borrow().weights()
    }
}
