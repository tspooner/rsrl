use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, Parameterised, VectorLFA, Projection, Projector, QFunction};
use crate::policies::{fixed::Greedy, Policy};

/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLambda<F, P> {
    pub fa_theta: Shared<F>,

    pub policy: Shared<P>,
    pub target: Greedy<F>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    trace: Trace,
}

impl<F, P> QLambda<F, P> {
    pub fn new<T1, T2>(
        fa_theta: Shared<F>,
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

impl<F, P: Algorithm> Algorithm for QLambda<F, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.borrow_mut().handle_terminal();
        self.target.handle_terminal();
    }
}

impl<S, M, P> OnlineLearner<S, P::Action> for QLambda<VectorLFA<M>, P>
where
    M: Projector<S>,
    P: Policy<S, Action = <Greedy<VectorLFA<M>> as Policy<S>>::Action>,
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

impl<S, F, P> Controller<S, P::Action> for QLambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = <Greedy<F> as Policy<S>>::Action>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, F, P> ValuePredictor<S> for QLambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = <Greedy<F> as Policy<S>>::Action>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        let a = self.target.sample(s);

        self.predict_qsa(s, a)
    }
}

impl<S, F, P> ActionValuePredictor<S, P::Action> for QLambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = <Greedy<F> as Policy<S>>::Action>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.fa_theta.borrow().evaluate_action(&s, a)
    }
}

impl<F: Parameterised, P> Parameterised for QLambda<F, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
