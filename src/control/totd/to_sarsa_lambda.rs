use core::{Algorithm, Predictor, Controller, Shared, Parameter, Vector, Matrix, Trace};
use domains::Transition;
use fa::{Approximator, Parameterised, MultiLFA, Projection, Projector, QFunction};
use policies::{Policy, FinitePolicy};
use std::marker::PhantomData;

/// True online variant of the SARSA(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
pub struct TOSARSALambda<S, M: Projector<S>, P: Policy<S>> {
    trace: Trace,
    q_old: f64,

    pub q_func: Shared<MultiLFA<S, M>>,
    pub policy: Shared<P>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M, P> TOSARSALambda<S, M, P>
where
    M: Projector<S>,
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

impl<S, M, P> Algorithm<S, usize> for TOSARSALambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = usize>,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.borrow().projector.project(s);

        let na = self.policy.borrow_mut().sample(ns);
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
    }
}

impl<S, M, P> Controller<S, usize> for TOSARSALambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = usize>,
{
    fn sample_target(&mut self, s: &S) -> usize { self.policy.borrow_mut().sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> usize { self.policy.borrow_mut().sample(s) }
}

impl<S, M, P> Predictor<S, usize> for TOSARSALambda<S, M, P>
where
    M: Projector<S>,
    P: FinitePolicy<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.policy.borrow_mut().probabilities(s))
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: usize) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}

impl<S, M, P> Parameterised for TOSARSALambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = usize>,
{
    fn weights(&self) -> Matrix<f64> {
        self.q_func.borrow().weights()
    }
}
