use core::*;
use domains::Transition;
use geometry::Matrix;
use fa::Parameterised;
use policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

pub struct BaselineREINFORCE<S, P, B> {
    pub policy: Shared<P>,
    pub baseline: Shared<B>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, P, B> BaselineREINFORCE<S, P, B> {
    pub fn new<T1, T2>(policy: Shared<P>, baseline: Shared<B>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        BaselineREINFORCE {
            policy,
            baseline,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, P, B> Algorithm for BaselineREINFORCE<S, P, B>
where
    P: Algorithm,
    B: Algorithm,
{
    fn step_hyperparams(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.borrow_mut().step_hyperparams();
        self.baseline.borrow_mut().step_hyperparams();
    }
}

impl<S, P, B> BatchLearner<S, P::Action> for BaselineREINFORCE<S, P, B>
where
    S: Clone,
    P: ParameterisedPolicy<S>,
    P::Action: Clone,
    B: BatchLearner<S, P::Action> + ActionValuePredictor<S, P::Action>,
{
    fn handle_batch(&mut self, batch: &[Transition<S, P::Action>]) {
        self.baseline.borrow_mut().handle_batch(batch);

        let mut ret = 0.0;

        for t in batch.into_iter().rev() {
            let s = t.from.state();
            let baseline = self.baseline.borrow_mut().predict_qsa(s, t.action.clone());

            ret = t.reward + self.gamma * ret;

            self.policy.borrow_mut().update(s, t.action.clone(), self.alpha * (ret - baseline));
        }
    }
}

impl<S, P, B> Controller<S, P::Action> for BaselineREINFORCE<S, P, B>
where
    P: ParameterisedPolicy<S>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, P, B> Parameterised for BaselineREINFORCE<S, P, B>
where
    P: ParameterisedPolicy<S>,
{
    fn weights(&self) -> Matrix<f64> {
        self.policy.borrow().weights()
    }
}
