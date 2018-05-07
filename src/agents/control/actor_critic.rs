use Parameter;
use agents::{Agent, Controller, TDPredictor};
use domains::Transition;
use fa::QFunction;
use policies::{Greedy, Policy};
use std::marker::PhantomData;

/// Regular gradient descent actor critic.
pub struct ActorCritic<S, Q, C, P>
where
    Q: QFunction<S>,
    C: TDPredictor<S>,
    P: Policy,
{
    pub q_func: Q, // actor
    pub critic: C,

    pub policy: P,

    pub beta: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, C, P> ActorCritic<S, Q, C, P>
where
    Q: QFunction<S>,
    C: TDPredictor<S>,
    P: Policy,
{
    pub fn new<T1, T2>(q_func: Q, critic: C, policy: P, beta: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        ActorCritic {
            q_func: q_func,
            critic: critic,

            policy: policy,

            beta: beta.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Clone, Q, C, P> Agent for ActorCritic<S, Q, C, P>
where
    Q: QFunction<S>,
    C: TDPredictor<S>,
    P: Policy,
{
    type Sample = Transition<S, usize>;

    fn handle_sample(&mut self, t: &Self::Sample) {
        let (s, ns) = (t.from.state(), t.to.state());
        let p_sample = (s.clone(), ns.clone(), t.reward);

        let td_error = self.critic.compute_td_error(&p_sample);

        self.critic.handle_td_error(&p_sample, td_error);
        self.q_func.update_action(s, t.action, self.beta * td_error);
    }

    fn handle_terminal(&mut self, _: &Self::Sample) {
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S: Clone, Q, C, P> Controller<S, usize> for ActorCritic<S, Q, C, P>
where
    Q: QFunction<S>,
    C: TDPredictor<S>,
    P: Policy,
{
    fn pi(&mut self, s: &S) -> usize {
        Greedy.sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize {
        self.policy
            .sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        p.sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }
}
