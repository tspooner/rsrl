use agents::{Controller, TDPredictor};
use domains::Transition;
use fa::QFunction;
use policies::{fixed::Greedy, Policy, FinitePolicy};
use std::marker::PhantomData;
use {Shared, Handler, Parameter};

/// Action-value actor-critic.
pub struct QAC<S, Q, C, P>
where
    Q: QFunction<S>,
    C: TDPredictor<S>,
    P: FinitePolicy<S>,
{
    pub actor: Shared<Q>, // actor
    pub critic: C,

    pub policy: P,
    pub target: Greedy<S>,

    pub beta: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, C, P> QAC<S, Q, C, P>
where
    Q: QFunction<S> + 'static,
    C: TDPredictor<S>,
    P: FinitePolicy<S>,
{
    pub fn new<T1, T2>(actor: Shared<Q>, critic: C, policy: P, beta: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        QAC {
            actor: actor.clone(),
            critic: critic,

            policy: policy,
            target: Greedy::new(actor),

            beta: beta.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Clone, Q, C, P> Handler<Transition<S, usize>> for QAC<S, Q, C, P>
where
    Q: QFunction<S>,
    C: TDPredictor<S>,
    P: FinitePolicy<S>,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let p_sample = t.into();
        let td_error = self.critic.compute_td_error(&p_sample);

        self.critic.handle_td_error(&p_sample, td_error);
        self.actor.borrow_mut().update_action(t.from.state(), t.action, self.beta * td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal(t);
    }
}

impl<S: Clone, Q, C, P> Controller<S, usize> for QAC<S, Q, C, P>
where
    Q: QFunction<S>,
    C: TDPredictor<S>,
    P: FinitePolicy<S>,
{
    fn pi(&mut self, s: &S) -> usize { self.target.sample(&s) }

    fn mu(&mut self, s: &S) -> usize { self.policy.sample(s) }
}
