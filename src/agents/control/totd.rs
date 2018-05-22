use agents::{Controller, memory::Trace};
use domains::Transition;
use fa::{Approximator, MultiLFA, Projection, Projector, QFunction};
use policies::{fixed::Greedy, FinitePolicy};
use std::marker::PhantomData;
use {Shared, Handler, Parameter};

/// True online variant of the Q(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
pub struct TOQLambda<S, M: Projector<S>, P: FinitePolicy<S>> {
    q_old: f64,
    trace: Trace,

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
            q_old: 0.0,
            trace: trace,

            q_func: q_func.clone(),

            policy: policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Controller<S, usize> for TOQLambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize { self.policy.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Handler<Transition<S, usize>> for TOQLambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.borrow().projector.project(s);

        let qs = self.q_func.borrow().evaluate_phi(&phi_s);
        let nqs = self.q_func.borrow().evaluate(&s).unwrap();
        let na = self.mu(ns);

        let td_error = t.reward + self.gamma * nqs[na] - qs[t.action];
        let phi_s = phi_s.expanded(self.q_func.borrow().projector.dim());

        let rate = self.trace.lambda.value() * self.gamma.value();
        let trace_update =
            (1.0 - rate * self.alpha.value() * self.trace.get().dot(&phi_s)) * phi_s.clone();

        if t.action == self.pi(s) {
            let rate = self.trace.lambda.value() * self.gamma.value();
            self.trace.decay(rate);
        } else {
            self.trace.decay(0.0);
        }

        self.trace.update(&trace_update);
        self.q_func.borrow_mut().update_action_phi(
            &Projection::Dense(self.trace.get()),
            t.action,
            self.alpha * (td_error + qs[t.action] - self.q_old),
        );
        self.q_func.borrow_mut().update_action_phi(
            &Projection::Dense(phi_s),
            t.action,
            self.alpha * (self.q_old - qs[t.action]),
        );

        self.q_old = nqs[na];
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);

        self.policy.handle_terminal(t);
        self.target.handle_terminal(t);
    }
}

/// True online variant of the SARSA(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
pub struct TOSARSALambda<S, M: Projector<S>, P: FinitePolicy<S>> {
    q_old: f64,
    trace: Trace,

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
            q_old: 0.0,
            trace: trace,

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
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.borrow().projector.project(s);

        let na = self.mu(ns);
        let nq = self.q_func.borrow().evaluate_action(s, na);
        let qsa = self.q_func.borrow().evaluate_action_phi(&phi_s, t.action);

        let td_error = t.reward + self.gamma * nq - qsa;
        let phi_s = phi_s.expanded(self.q_func.borrow().projector.dim());

        let rate = self.trace.lambda.value() * self.gamma.value();
        let trace_update =
            (1.0 - rate * self.alpha.value() * self.trace.get().dot(&phi_s)) * phi_s.clone();

        self.trace.decay(rate);
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

        self.q_old = nq;
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);

        self.policy.handle_terminal(t);
    }
}
