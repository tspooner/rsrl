use core::*;
use domains::Transition;
use fa::{Parameterised, QFunction};
use policies::{fixed::Greedy, Policy};

/// Persistent Advantage Learning
///
/// # References
/// - Bellemare, Marc G., et al. "Increasing the Action Gap: New Operators for
/// Reinforcement Learning." AAAI. 2016.
pub struct PAL<S, Q, P> {
    pub q_func: Shared<Q>,

    pub policy: Shared<P>,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<S, Q, P> PAL<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: Policy<S>,
{
    pub fn new<T1, T2>(q_func: Shared<Q>, policy: Shared<P>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        PAL {
            q_func: q_func.clone(),

            policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<S, Q, P: Algorithm> Algorithm for PAL<S, Q, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.borrow_mut().handle_terminal();
        self.target.handle_terminal();
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<S> as Policy<S>>::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.predict_qs(s);

        let residual = if t.terminated() {
            t.reward - qs[t.action]

        } else {
            let nqs = self.predict_qs(ns);
            let a_star = self.sample_target(s);
            let na_star = self.sample_target(ns);

            let td_error = t.reward + self.gamma * nqs[a_star] - qs[t.action];
            let al_error = td_error - self.alpha * (qs[a_star] - qs[t.action]);

            al_error.max(td_error - self.alpha * (nqs[na_star] - nqs[t.action]))
        };

        self.q_func.borrow_mut().update_action(s, t.action, self.alpha * residual);
    }
}

impl<S, Q, P> Controller<S, P::Action> for PAL<S, Q, P>
where
    P: Policy<S, Action = <Greedy<S> as Policy<S>>::Action>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, Q, P> ValuePredictor<S> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<S> as Policy<S>>::Action>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        let a = self.target.sample(s);

        self.predict_qsa(s, a)
    }
}

impl<S, Q, P> ActionValuePredictor<S, P::Action> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<S> as Policy<S>>::Action>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}

impl<S, Q, P> Parameterised for PAL<S, Q, P>
where
    Q: Parameterised,
{
    fn weights(&self) -> Matrix<f64> {
        self.q_func.borrow().weights()
    }
}
