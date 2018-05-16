use agents::{Controller, memory::Trace};
use domains::Transition;
use fa::{Approximator, MultiLFA, Projection, Projector, QFunction};
use policies::{Greedy, Policy};
use std::marker::PhantomData;
use {Handler, Parameter, Vector};

/// True online variant of the Q(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
pub struct TOQLambda<S, M: Projector<S>, P: Policy<[f64], usize>> {
    trace: Trace,
    q_old: f64,

    pub q_func: MultiLFA<S, M>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M, P> TOQLambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<[f64], usize>,
{
    pub fn new<T1, T2>(
        trace: Trace,
        q_func: MultiLFA<S, M>,
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

            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: Policy<[f64], usize>> Controller<S, usize> for TOQLambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize {
        let qs: Vector<f64> = self.q_func.evaluate(s).unwrap();

        self.policy.sample(qs.as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }
}

impl<S, M: Projector<S>, P: Policy<[f64], usize>> Handler<Transition<S, usize>> for TOQLambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.projector.project(s);
        let phi_ns = self.q_func.projector.project(ns);

        let qs = self.q_func.evaluate_phi(&phi_s);
        let nqs = self.q_func.evaluate_phi(&phi_ns);
        let na = Greedy.sample(nqs.as_slice().unwrap());

        let td_error = t.reward + self.gamma * nqs[na] - qs[t.action];
        let phi_s = phi_s.expanded(self.q_func.projector.dim());

        let rate = self.trace.lambda.value() * self.gamma.value();
        let trace_update =
            (1.0 - rate * self.alpha.value() * self.trace.get().dot(&phi_s)) * phi_s.clone();

        if t.action == Greedy.sample(qs.as_slice().unwrap()) {
            let rate = self.trace.lambda.value() * self.gamma.value();
            self.trace.decay(rate);
        } else {
            self.trace.decay(0.0);
        }

        self.trace.update(&trace_update);
        self.q_func.update_action_phi(
            &Projection::Dense(self.trace.get()),
            t.action,
            self.alpha * (td_error + qs[t.action] - self.q_old),
        );
        self.q_func.update_action_phi(
            &Projection::Dense(phi_s),
            t.action,
            self.alpha * (self.q_old - qs[t.action]),
        );

        self.q_old = nqs[na];
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
        self.policy.handle_terminal();
    }
}

/// True online variant of the SARSA(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
pub struct TOSARSALambda<S, M: Projector<S>, P: Policy<[f64], usize>> {
    trace: Trace,
    q_old: f64,

    pub q_func: MultiLFA<S, M>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M, P> TOSARSALambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<[f64], usize>,
{
    pub fn new<T1, T2>(
        trace: Trace,
        q_func: MultiLFA<S, M>,
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

impl<S, M: Projector<S>, P: Policy<[f64], usize>> Controller<S, usize> for TOSARSALambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize {
        let qs: Vector<f64> = self.q_func.evaluate(s).unwrap();

        self.policy.sample(qs.as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }
}

impl<S, M: Projector<S>, P: Policy<[f64], usize>> Handler<Transition<S, usize>> for TOSARSALambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.projector.project(s);
        let phi_ns = self.q_func.projector.project(ns);

        let qsa = self.q_func.evaluate_action_phi(&phi_s, t.action);
        let nqs = self.q_func.evaluate_phi(&phi_ns);
        let na = self.policy.sample(nqs.as_slice().unwrap());

        let td_error = t.reward + self.gamma * nqs[na] - qsa;
        let phi_s = phi_s.expanded(self.q_func.projector.dim());

        let rate = self.trace.lambda.value() * self.gamma.value();
        let trace_update =
            (1.0 - rate * self.alpha.value() * self.trace.get().dot(&phi_s)) * phi_s.clone();

        self.trace.decay(rate);
        self.trace.update(&trace_update);

        self.q_func.update_action_phi(
            &Projection::Dense(self.trace.get()),
            t.action,
            self.alpha * (td_error + qsa - self.q_old),
        );
        self.q_func.update_action_phi(
            &Projection::Dense(phi_s),
            t.action,
            self.alpha * (self.q_old - qsa),
        );

        self.q_old = nqs[na];
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
        self.policy.handle_terminal();
    }
}
