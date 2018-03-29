use agents::memory::Trace;
use agents::{Agent, Controller};
use domains::Transition;
use fa::{Approximator, MultiLinear, Projection, Projector, QFunction};
use geometry::{ActionSpace, Space};
use policies::{Greedy, Policy};
use std::marker::PhantomData;
use {Parameter, Vector};

/// True online variant of the Q(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
pub struct TOQLambda<S, M: Projector<S>, P: Policy> {
    trace: Trace,
    q_old: f64,

    pub q_func: MultiLinear<S, M>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M, P> TOQLambda<S, M, P>
where
    M: Projector<S>,
    P: Policy,
{
    pub fn new<T1, T2>(
        trace: Trace,
        q_func: MultiLinear<S, M>,
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

impl<S, M: Projector<S>, P: Policy> Controller<S, <ActionSpace as Space>::Repr>
    for TOQLambda<S, M, P>
{
    fn pi(&mut self, s: &S) -> usize {
        let qs: Vector<f64> = self.q_func.evaluate(s).unwrap();

        self.policy.sample(qs.as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        let qs: Vector<f64> = self.q_func.evaluate(s).unwrap();

        p.sample(qs.as_slice().unwrap())
    }
}

impl<S, M: Projector<S>, P: Policy> Agent for TOQLambda<S, M, P> {
    type Sample = Transition<S, <ActionSpace as Space>::Repr>;

    fn handle_sample(&mut self, t: &Transition<S, <ActionSpace as Space>::Repr>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.projector.project(s);
        let phi_ns = self.q_func.projector.project(ns);

        let qs = self.q_func.evaluate_phi(&phi_s);
        let nqs = self.q_func.evaluate_phi(&phi_ns);
        let na = Greedy.sample(nqs.as_slice().unwrap());

        let td_error = t.reward + self.gamma * nqs[na] - qs[a];
        let phi_s = phi_s.expanded(self.q_func.projector.span());

        let rate = self.trace.lambda.value() * self.gamma.value();
        let trace_update =
            (1.0 - rate * self.alpha.value() * self.trace.get().dot(&phi_s)) * phi_s.clone();

        if a == Greedy.sample(qs.as_slice().unwrap()) {
            let rate = self.trace.lambda.value() * self.gamma.value();
            self.trace.decay(rate);
        } else {
            self.trace.decay(0.0);
        }

        self.trace.update(&trace_update);
        self.q_func.update_action_phi(
            &Projection::Dense(self.trace.get()),
            a,
            self.alpha * (td_error + qs[a] - self.q_old),
        );
        self.q_func.update_action_phi(
            &Projection::Dense(phi_s),
            a,
            self.alpha * (self.q_old - qs[a]),
        );

        self.q_old = nqs[na];
    }

    fn handle_terminal(&mut self, _: &Self::Sample) {
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
pub struct TOSARSALambda<S, M: Projector<S>, P: Policy> {
    trace: Trace,
    q_old: f64,

    pub q_func: MultiLinear<S, M>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M, P> TOSARSALambda<S, M, P>
where
    M: Projector<S>,
    P: Policy,
{
    pub fn new<T1, T2>(
        trace: Trace,
        q_func: MultiLinear<S, M>,
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

impl<S, M: Projector<S>, P: Policy> Controller<S, <ActionSpace as Space>::Repr>
    for TOSARSALambda<S, M, P>
{
    fn pi(&mut self, s: &S) -> usize {
        let qs: Vector<f64> = self.q_func.evaluate(s).unwrap();

        self.policy.sample(qs.as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        let qs: Vector<f64> = self.q_func.evaluate(s).unwrap();

        p.sample(qs.as_slice().unwrap())
    }
}

impl<S, M: Projector<S>, P: Policy> Agent for TOSARSALambda<S, M, P> {
    type Sample = Transition<S, <ActionSpace as Space>::Repr>;

    fn handle_sample(&mut self, t: &Transition<S, <ActionSpace as Space>::Repr>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.projector.project(s);
        let phi_ns = self.q_func.projector.project(ns);

        let qsa = self.q_func.evaluate_action_phi(&phi_s, a);
        let nqs = self.q_func.evaluate_phi(&phi_ns);
        let na = self.policy.sample(nqs.as_slice().unwrap());

        let td_error = t.reward + self.gamma * nqs[na] - qsa;
        let phi_s = phi_s.expanded(self.q_func.projector.span());

        let rate = self.trace.lambda.value() * self.gamma.value();
        let trace_update =
            (1.0 - rate * self.alpha.value() * self.trace.get().dot(&phi_s)) * phi_s.clone();

        self.trace.decay(rate);
        self.trace.update(&trace_update);

        self.q_func.update_action_phi(
            &Projection::Dense(self.trace.get()),
            a,
            self.alpha * (td_error + qsa - self.q_old),
        );
        self.q_func.update_action_phi(
            &Projection::Dense(phi_s),
            a,
            self.alpha * (self.q_old - qsa),
        );

        self.q_old = nqs[na];
    }

    fn handle_terminal(&mut self, _: &Self::Sample) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
        self.policy.handle_terminal();
    }
}
