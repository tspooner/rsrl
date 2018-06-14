use core::{Controller, Predictor, Handler, Shared, Parameter, Vector, Trace};
use domains::Transition;
use fa::{Approximator, MultiLFA, Projection, Projector, QFunction};
use policies::FinitePolicy;
use std::marker::PhantomData;

/// True online variant of the SARSA(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
pub struct TOSARSALambda<S, M: Projector<S>, P: FinitePolicy<S>> {
    trace: Trace,
    q_old: f64,

    pub q_func: Shared<MultiLFA<S, M>>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M, P> TOSARSALambda<S, M, P>
where
    M: Projector<S>,
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
        TOSARSALambda {
            trace: trace,
            q_old: 0.0,

            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Controller<S, usize> for TOSARSALambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize { self.policy.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Handler<Transition<S, usize>> for TOSARSALambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.borrow().projector.project(s);

        let na = self.policy.sample(ns);
        let qsa = self.q_func.borrow().evaluate_action_phi(&phi_s, a);
        let nqa = self.q_func.borrow().evaluate_action(ns, na);

        let td_error = t.reward + self.gamma * nqa - qsa;
        let phi_s = phi_s.expanded(self.q_func.borrow().projector.dim());

        let rate = self.trace.lambda.value() * self.gamma.value();
        let trace_update =
            (1.0 - rate * self.alpha.value() * self.trace.get().dot(&phi_s)) * phi_s.clone();

        self.trace.decay(rate);
        self.trace.update(&trace_update);

        self.q_func.borrow_mut().update_action_phi(
            &Projection::Dense(self.trace.get()),
            a,
            self.alpha * (td_error + qsa - self.q_old),
        );
        self.q_func.borrow_mut().update_action_phi(
            &Projection::Dense(phi_s),
            a,
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

impl<S, M: Projector<S>, P: FinitePolicy<S>> Predictor<S, usize> for TOSARSALambda<S, M, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.policy.probabilities(s))
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: usize) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}
