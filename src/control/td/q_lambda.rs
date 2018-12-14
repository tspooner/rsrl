use core::*;
use domains::Transition;
use fa::{Approximator, Parameterised, MultiLFA, Projection, Projector, QFunction};
use policies::{fixed::Greedy, Policy};

/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLambda<S, M: Projector<S>, P> {
    pub fa_theta: Shared<MultiLFA<S, M>>,

    pub policy: Shared<P>,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    trace: Trace,
}

impl<S, M: Projector<S>, P> QLambda<S, M, P>
where
    S: 'static,
    M: Projector<S> + 'static,
{
    pub fn new<T1, T2>(
        fa_theta: Shared<MultiLFA<S, M>>,
        policy: Shared<P>,
        trace: Trace,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        QLambda {
            fa_theta: fa_theta.clone(),

            policy,
            target: Greedy::new(fa_theta),

            alpha: alpha.into(),
            gamma: gamma.into(),

            trace,
        }
    }
}

impl<S, M, P> Algorithm for QLambda<S, M, P>
where
    M: Projector<S>,
    P: Algorithm,
{
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.borrow_mut().handle_terminal();
        self.target.handle_terminal();
    }
}

impl<S, M, P> OnlineLearner<S, P::Action> for QLambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = <Greedy<S> as Policy<S>>::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.borrow().projector.project(s);
        let qsa = self.fa_theta.borrow().evaluate_action_phi(&phi_s, t.action);

        // Update trace:
        let n_bases = self.fa_theta.borrow().projector.dim();
        let decay_rate = if t.action == self.target.sample(s) {
            self.trace.lambda.value() * self.gamma.value()
        } else {
            0.0
        };

        self.trace.decay(decay_rate);
        self.trace.update(&phi_s.expanded(n_bases));

        // Update weight vectors:
        let z = self.trace.get();
        let residual = if t.terminated() {
            self.trace.decay(0.0);

            t.reward - qsa
        } else {
            let na = self.target.sample(&ns);
            let nqsna = self.fa_theta.borrow().evaluate_action(ns, na);

            t.reward + self.gamma * nqsna - qsa
        };

        self.fa_theta.borrow_mut().update_action_phi(
            &Projection::Dense(z), t.action,
            self.alpha * residual,
        );
    }
}

impl<S, M, P> Controller<S, P::Action> for QLambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = <Greedy<S> as Policy<S>>::Action>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, M, P> ValuePredictor<S> for QLambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = <Greedy<S> as Policy<S>>::Action>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        let a = self.target.sample(s);

        self.predict_qsa(s, a)
    }
}

impl<S, M, P> ActionValuePredictor<S, P::Action> for QLambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = <Greedy<S> as Policy<S>>::Action>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.fa_theta.borrow().evaluate_action(&s, a)
    }
}

impl<S, M, P> Parameterised for QLambda<S, M, P>
where
    M: Projector<S>,
    P: Parameterised,
{
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
