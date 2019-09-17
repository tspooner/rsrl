use crate::{
    Algorithm, OnlineLearner, Parameter, Shared, make_shared,
    control::Controller,
    domains::Transition,
    fa::{
        Parameterised, Weights, WeightsView, WeightsViewMut,
        StateActionFunction, EnumerableStateActionFunction,
        DifferentiableStateActionFunction,
        traces::Trace,
    },
    policies::{Policy, FinitePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
};
use rand::Rng;

/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
#[derive(Parameterised, Serialize, Deserialize)]
pub struct QLambda<F, P, T> {
    #[weights] pub fa_theta: F,

    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub lambda: Parameter,

    trace: T,
}

impl<F, P, T> QLambda<Shared<F>, P, T> {
    pub fn new<T1, T2, T3>(
        fa_theta: F,
        policy: P,
        trace: T,
        alpha: T1,
        gamma: T2,
        lambda: T3,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        let fa_theta = make_shared(fa_theta);

        QLambda {
            fa_theta: fa_theta.clone(),

            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),
            lambda: lambda.into(),

            trace,
        }
    }
}

impl<F, P: Algorithm, T: Algorithm> Algorithm for QLambda<F, P, T> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
        self.lambda = self.lambda.step();

        self.policy.handle_terminal();
        self.trace.handle_terminal();
    }
}

impl<S, F, P, T> OnlineLearner<S, P::Action> for QLambda<F, P, T>
where
    F: EnumerableStateActionFunction<S> + DifferentiableStateActionFunction<S, usize>,
    P: FinitePolicy<S>,
    T: Trace<F::Gradient>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let qsa = self.fa_theta.evaluate(s, &t.action);

        // Update trace:
        self.trace.scale(if t.action == self.fa_theta.find_max(s).0 {
            self.lambda.value() * self.gamma.value()
        } else {
            0.0
        });
        self.trace.update(&self.fa_theta.grad(s, &t.action));

        // Update weight vectors:
        if t.terminated() {
            self.fa_theta.update_grad_scaled(self.trace.deref(), self.alpha * (t.reward - qsa));
            self.trace.reset();
        } else {
            let ns = t.to.state();
            let (_, nqs_max) = self.fa_theta.find_max(ns);
            let residual = t.reward + self.gamma * nqs_max - qsa;

            self.fa_theta.update_grad_scaled(self.trace.deref(), self.alpha * residual);
        }
    }
}

impl<S, F, P, T> Controller<S, P::Action> for QLambda<F, P, T>
where
    F: EnumerableStateActionFunction<S, Output = f64>,
    P: FinitePolicy<S>,
{
    fn sample_target(&self, _: &mut impl Rng, s: &S) -> P::Action {
        self.fa_theta.find_max(s).0
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}

impl<S, F, P, T> ValuePredictor<S> for QLambda<F, P, T>
where
    F: EnumerableStateActionFunction<S, Output = f64>,
    P: Policy<S>,
{
    fn predict_v(&self, s: &S) -> f64 { self.fa_theta.find_max(s).1 }
}

impl<S, F, P, T> ActionValuePredictor<S, P::Action> for QLambda<F, P, T>
where
    F: StateActionFunction<S, P::Action, Output = f64>,
    P: Policy<S>,
{
    fn predict_qsa(&self, s: &S, a: P::Action) -> f64 {
        self.fa_theta.evaluate(s, &a)
    }
}
