use agents::{Controller, TDPredictor};
use domains::Transition;
use fa::QFunction;
use policies::{Greedy, Policy};
use std::marker::PhantomData;
use {Handler, Parameter};

/// Regular gradient descent actor critic.
pub struct ActorCritic<S, Q, C, P>
where
    Q: QFunction<S>,
    C: TDPredictor<S>,
    P: Policy<[f64], usize>,
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
    P: Policy<[f64], usize>,
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

impl<S: Clone, Q, C, P> Handler for ActorCritic<S, Q, C, P>
where
    Q: QFunction<S>,
    C: TDPredictor<S>,
    P: Policy<[f64], usize>,
{
    type Sample = Transition<S, usize>;

    fn handle_sample(&mut self, t: &Self::Sample) {
        let p_sample = t.into();
        let td_error = self.critic.compute_td_error(&p_sample);

        self.critic.handle_td_error(&p_sample, td_error);
        self.q_func.update_action(t.from.state(), t.action, self.beta * td_error);
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
    P: Policy<[f64], usize>,
{
    fn pi(&mut self, s: &S) -> usize {
        Greedy.sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize {
        self.policy
            .sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }
}
