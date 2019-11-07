use crate::{
    OnlineLearner, Shared, make_shared,
    control::Controller,
    domains::Transition,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction, StateActionFunction, EnumerableStateActionFunction,
        linear::{LinearStateFunction, LinearStateActionFunction},
    },
    policies::{Greedy, Policy, EnumerablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
};
use rand::{thread_rng, Rng};

// TODO: Extract prediction component GQ / GQ(lambda) into seperate implementations.

/// Greedy GQ control algorithm.
///
/// Maei, Hamid R., et al. "Toward off-policy learning control with function
/// approximation." Proceedings of the 27th International Conference on Machine
/// Learning (ICML-10). 2010.
#[derive(Parameterised)]
pub struct GreedyGQ<Q, W, PB> {
    #[weights] pub fa_q: Q,
    pub fa_w: W,

    pub target_policy: Greedy<Q>,
    pub behaviour_policy: PB,

    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl<Q, W, PB> GreedyGQ<Shared<Q>, W, PB> {
    pub fn new(
        fa_q: Q,
        fa_w: W,
        behaviour_policy: PB,
        alpha: f64,
        beta: f64,
        gamma: f64,
    ) -> Self {
        let fa_q = make_shared(fa_q);

        GreedyGQ {
            fa_q: fa_q.clone(),
            fa_w,

            target_policy: Greedy::new(fa_q),
            behaviour_policy,

            alpha,
            beta,
            gamma,
        }
    }
}

impl<S, Q, W, PB> OnlineLearner<S, PB::Action> for GreedyGQ<Q, W, PB>
where
    Q: EnumerableStateActionFunction<S> + LinearStateActionFunction<S, usize>,
    W: StateFunction<S, Output = f64> + LinearStateFunction<S>,
    PB: EnumerablePolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, PB::Action>) {
        let s = t.from.state();

        let phi_s_w = self.fa_w.features(s);
        let phi_s_q = self.fa_q.features(s, &t.action);

        let qsa = self.fa_q.evaluate_features(&phi_s_q, &t.action);
        let estimate = self.fa_w.evaluate_features(&phi_s_w);

        if t.terminated() {
            let residual = t.reward - qsa;

            self.fa_w.update_features(&phi_s_w, self.alpha * self.beta * (residual - estimate));
            self.fa_q.update_features(&phi_s_q, &t.action, self.alpha * residual);
        } else {
            let ns = t.to.state();
            let na = self.sample_target(&mut thread_rng(), ns);
            let phi_ns_q = self.fa_q.features(ns, &na);

            let residual =
                t.reward + self.gamma * self.fa_q.evaluate_features(&phi_ns_q, &na) - qsa;

            let update_q = phi_s_q.combine(&phi_ns_q, |x, y| {
                residual * x - estimate * self.gamma * y
            });

            self.fa_w.update_features(&phi_s_w, self.alpha * self.beta * (residual - estimate));
            self.fa_q.update_features(&update_q, &t.action, self.alpha);
        }
    }
}

impl<S, Q, W, PB> ValuePredictor<S> for GreedyGQ<Q, W, PB>
where
    Q: StateActionFunction<S, <Greedy<Q> as Policy<S>>::Action, Output = f64>,
    Greedy<Q>: Policy<S>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_q.evaluate(s, &self.target_policy.mpa(s))
    }
}

impl<S, Q, W, PB> ActionValuePredictor<S, <Greedy<Q> as Policy<S>>::Action> for GreedyGQ<Q, W, PB>
where
    Q: StateActionFunction<S, <Greedy<Q> as Policy<S>>::Action, Output = f64>,
    Greedy<Q>: Policy<S>,
{
    fn predict_q(&self, s: &S, a: &<Greedy<Q> as Policy<S>>::Action) -> f64 {
        self.fa_q.evaluate(s, a)
    }
}

impl<S, Q, W, PB> Controller<S, PB::Action> for GreedyGQ<Q, W, PB>
where
    Q: EnumerableStateActionFunction<S>,
    PB: EnumerablePolicy<S>,
{
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> PB::Action {
        self.target_policy.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> PB::Action {
        self.behaviour_policy.sample(rng, s)
    }
}
