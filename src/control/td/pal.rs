use core::{Algorithm, Controller, Predictor, Shared, Parameter, Vector, Matrix};
use domains::Transition;
use fa::{Parameterised, QFunction};
use policies::{fixed::Greedy, Policy};
use std::marker::PhantomData;

/// Persistent Advantage Learning
///
/// # References
/// - Bellemare, Marc G., et al. "Increasing the Action Gap: New Operators for
/// Reinforcement Learning." AAAI. 2016.
pub struct PAL<S, Q: QFunction<S>, P: Policy<S>> {
    pub q_func: Shared<Q>,

    pub policy: Shared<P>,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
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

            phantom: PhantomData,
        }
    }
}

impl<S, Q, P> Algorithm<S, P::Action> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = usize>,
{
    fn handle_sample(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.predict_qs(s);
        let nqs = self.predict_qs(s);

        let qa_star = qs[self.sample_target(&s)];

        let td_error = t.reward + self.gamma * nqs[self.sample_target(&ns)] - qs[t.action];
        let al_error = td_error - self.alpha * (qa_star - qs[t.action]);
        let pal_error = al_error.max(td_error - self.alpha * (qa_star - nqs[t.action]));

        self.q_func.borrow_mut().update_action(s, t.action, self.alpha * pal_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, P::Action>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.target.handle_terminal(t);
        self.policy.borrow_mut().handle_terminal(t);
    }
}

impl<S, Q, P> Controller<S, P::Action> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = usize>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, Q, P> Predictor<S, P::Action> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = usize>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.q_func.borrow().evaluate(s).unwrap()[self.target.sample(s)]
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}

impl<S, Q, P> Parameterised for PAL<S, Q, P>
where
    Q: QFunction<S> + Parameterised,
    P: Policy<S, Action = usize>,
{
    fn weights(&self) -> Matrix<f64> {
        self.q_func.borrow().weights()
    }
}
