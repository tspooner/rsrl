use core::{Algorithm, Controller, Predictor, Parameter, Shared};
use domains::Transition;
use geometry::Matrix;
use fa::Parameterised;
use policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

pub struct REINFORCE<S, P: Policy<S>> {
    pub policy: Shared<P>,
    pub cache: Vec<(S, P::Action, f64)>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, P: ParameterisedPolicy<S>> REINFORCE<S, P> {
    pub fn new<T1, T2>(policy: Shared<P>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        REINFORCE {
            policy,

            cache: vec![],

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }

    pub fn propagate(&mut self) {
        let mut ret = 0.0;

        for (s, a, r) in self.cache.drain(0..).rev() {
            ret = r + self.gamma * ret;

            self.policy.borrow_mut().update(&s, a, self.alpha * ret);
        }
    }
}

impl<S: Clone, P: ParameterisedPolicy<S>> Algorithm<S, P::Action> for REINFORCE<S, P>
where
    P::Action: Clone,
{
    fn handle_sample(&mut self, t: &Transition<S, P::Action>) {
        self.cache.push((t.from.state().clone(), t.action.clone(), t.reward));
    }

    fn handle_terminal(&mut self, _: &Transition<S, P::Action>) {
        {
            self.propagate();
        }

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S: Clone, P: ParameterisedPolicy<S>> Controller<S, P::Action> for REINFORCE<S, P>
where
    P::Action: Clone,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, P: Policy<S> + Parameterised> Parameterised for REINFORCE<S, P> {
    fn weights(&self) -> Matrix<f64> {
        self.policy.borrow().weights()
    }
}


pub struct BaselineREINFORCE<S, P: Policy<S>, C: Predictor<S, P::Action>> {
    pub policy: Shared<P>,
    pub baseline: C,
    pub cache: Vec<(S, P::Action, f64)>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, P, C> BaselineREINFORCE<S, P, C>
where
    P: ParameterisedPolicy<S>,
    C: Predictor<S, P::Action>,
{
    pub fn new<T1, T2>(policy: Shared<P>, baseline: C, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        BaselineREINFORCE {
            policy,
            baseline,

            cache: vec![],

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }

    pub fn propagate(&mut self) {
        let mut ret = 0.0;

        for (s, a, r) in self.cache.drain(0..).rev() {
            let baseline = self.baseline.predict_v(&s);

            ret = r + self.gamma * ret;

            self.policy.borrow_mut().update(&s, a, self.alpha * (ret - baseline));
        }
    }
}

impl<S: Clone, P, C> Algorithm<S, P::Action> for BaselineREINFORCE<S, P, C>
where
    P: ParameterisedPolicy<S>,
    P::Action: Clone,
    C: Predictor<S, P::Action>,
{
    fn handle_sample(&mut self, t: &Transition<S, P::Action>) {
        self.baseline.handle_sample(t);
        self.cache.push((t.from.state().clone(), t.action.clone(), t.reward));
    }

    fn handle_terminal(&mut self, t: &Transition<S, P::Action>) {
        self.baseline.handle_terminal(t);

        self.propagate();

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S: Clone, P, C> Controller<S, P::Action> for BaselineREINFORCE<S, P, C>
where
    P: ParameterisedPolicy<S>,
    P::Action: Clone,
    C: Predictor<S, P::Action>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, P, C> Parameterised for BaselineREINFORCE<S, P, C>
where
    P: ParameterisedPolicy<S>,
    P::Action: Clone,
    C: Predictor<S, P::Action>,
{
    fn weights(&self) -> Matrix<f64> {
        self.policy.borrow().weights()
    }
}