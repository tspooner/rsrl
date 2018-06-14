use core::{Controller, Predictor, Handler, Shared, Parameter, Vector};
use domains::Transition;
use fa::QFunction;
use policies::{fixed::Greedy, Policy, FinitePolicy};
use std::marker::PhantomData;

/// Persistent Advantage Learning
///
/// # References
/// - Bellemare, Marc G., et al. "Increasing the Action Gap: New Operators for
/// Reinforcement Learning." AAAI. 2016.
pub struct PAL<S, Q: QFunction<S>, P: FinitePolicy<S>> {
    pub q_func: Shared<Q>,

    pub policy: P,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> PAL<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    pub fn new<T1, T2>(q_func: Shared<Q>, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        PAL {
            q_func: q_func.clone(),

            policy: policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, Q, P> Handler<Transition<S, usize>> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.borrow().evaluate(s).unwrap();
        let nqs = self.q_func.borrow().evaluate(ns).unwrap();

        let qa_star = qs[self.target.sample(&s)];

        let td_error = t.reward + self.gamma * nqs[self.target.sample(&ns)] - qs[t.action];
        let al_error = td_error - self.alpha * (qa_star - qs[t.action]);
        let pal_error = al_error.max(td_error - self.alpha * (qa_star - nqs[t.action]));

        self.q_func.borrow_mut().update_action(s, t.action, self.alpha * pal_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal(t);
    }
}

impl<S, Q, P> Controller<S, usize> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn pi(&mut self, s: &S) -> usize { self.target.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.policy.sample(s) }
}

impl<S, Q, P> Predictor<S, usize> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.q_func.borrow().evaluate(s).unwrap()[self.target.sample(s)]
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: usize) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}
