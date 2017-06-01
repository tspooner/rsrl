use Parameter;
use fa::QFunction;
use utils::dot;
use agents::ControlAgent;
use domains::Transition;
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};
use std::marker::PhantomData;


/// Watkins' classical off policy temporal difference control algorithm.
///
/// C. J. C. H. Watkins and P. Dayan, “Q-learning,” Mach. Learn., vol. 8, no. 3–4, pp. 279–292,
/// 1992.
pub struct QLearning<S: Space, Q: QFunction<S>, P: Policy>
{
    q_func: Q,
    policy: P,

    alpha: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, Q, P> QLearning<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    pub fn new<T1, T2>(q_func: Q, policy: P,
                       alpha: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        QLearning {
            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, Q, P> ControlAgent<S, ActionSpace> for QLearning<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn pi_target(&mut self, s: &S::Repr) -> usize {
        Greedy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s);
        let nqs = self.q_func.evaluate(ns);

        let a = t.action;
        let na = Greedy.sample(nqs.as_slice());

        let td_error = self.alpha*(t.reward + self.gamma*nqs[na] - qs[a]);

        self.q_func.update_action(s, a, td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}


/// Classical on policy temporal difference control algorithm.
pub struct SARSA<S: Space, Q: QFunction<S>, P: Policy>
{
    q_func: Q,
    policy: P,

    alpha: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>
}

impl<S: Space, Q, P> SARSA<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    pub fn new<T1, T2>(q_func: Q, policy: P,
                       alpha: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        SARSA {
            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, Q, P> ControlAgent<S, ActionSpace> for SARSA<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn pi_target(&mut self, s: &S::Repr) -> usize {
        Greedy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s);
        let nqs = self.q_func.evaluate(ns);

        let a = t.action;
        let na = self.policy.sample(nqs.as_slice());

        let td_error = self.alpha*(t.reward + self.gamma*nqs[na] - qs[a]);

        self.q_func.update_action(s, a, td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}


/// Expected SARSA.
pub struct ExpectedSARSA<S: Space, Q: QFunction<S>, P: Policy>
{
    q_func: Q,
    policy: P,

    alpha: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>
}

impl<S: Space, Q, P> ExpectedSARSA<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    pub fn new<T1, T2>(q_func: Q, policy: P,
                       alpha: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        ExpectedSARSA {
            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, Q, P> ControlAgent<S, ActionSpace> for ExpectedSARSA<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn pi_target(&mut self, s: &S::Repr) -> usize {
        Greedy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s);
        let nqs = self.q_func.evaluate(ns);

        let a = t.action;

        let exp_nqs = dot(&nqs, &self.policy.probabilities(nqs.as_slice()));
        let td_error = self.alpha*(t.reward + self.gamma*exp_nqs - qs[a]);

        self.q_func.update_action(s, a, td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}
