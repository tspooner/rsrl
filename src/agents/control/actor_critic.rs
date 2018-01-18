use Parameter;
use agents::{Agent, LinearAgent, Controller, Predictor, TDPredictor};
use domains::Transition;
use fa::{Function, QFunction, Projector, Projection, Linear};
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};
use ndarray::Array2;
use std::marker::PhantomData;


/// Regular gradient descent actor critic.
pub struct ActorCritic<S: Space, Q, C, P>
    where Q: QFunction<S>,
          C: TDPredictor<S>,
          P: Policy
{
    pub q_func: Q,
    pub critic: C,

    pub policy: P,

    pub beta: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, Q, C, P> ActorCritic<S, Q, C, P>
    where Q: QFunction<S>,
          C: TDPredictor<S>,
          P: Policy
{
    pub fn new<T1, T2>(q_func: Q, critic: C, policy: P, beta: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
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

impl<S: Space, Q, C, P> Agent<S> for ActorCritic<S, Q, C, P>
    where Q: QFunction<S>,
          C: TDPredictor<S>,
          P: Policy
{
    type Sample = Transition<S, ActionSpace>;

    fn handle_sample(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let td_error = self.critic.compute_error(&(s.clone(), ns.clone(), t.reward));

        self.critic.handle_error(&(s.clone(), ns.clone(), t.reward), td_error);
        self.q_func.update_action(s, t.action, self.beta*td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S: Space, M, C, P> LinearAgent<S> for ActorCritic<S, Linear<S, M>, C, P>
    where M: Projector<S>,
          C: TDPredictor<S>,
          P: Policy
{
    fn weights(&self) -> Array2<f64> {
        self.q_func.weights.clone()
    }
}

impl<S: Space, Q, C, P> Controller<S, ActionSpace> for ActorCritic<S, Q, C, P>
    where Q: QFunction<S>,
          C: TDPredictor<S>,
          P: Policy
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        Greedy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn mu(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize {
        p.sample(self.q_func.evaluate(s).as_slice())
    }
}

// TODO: Implement NAC
// TODO: Implement Episodic NAC
