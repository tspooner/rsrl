use Parameter;
use fa::{VFunction, QFunction, Linear};
use agents::ControlAgent;
use domains::Transition;
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};
use std::marker::PhantomData;


/// Greedy GQ control algorithm
///
/// Maei, Hamid R., et al. "Toward off-policy learning control with function approximation."
/// Proceedings of the 27th International Conference on Machine Learning (ICML-10). 2010.
pub struct GreedyGQ<S: Space, Q: QFunction<S>, V: VFunction<S>, P: Policy> {
    q_func: Q,
    v_func: V,

    policy: P,

    alpha: Parameter,
    beta: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>
}

impl<S: Space, Q, V, P> GreedyGQ<S, Q, V, P>
    where Q: QFunction<S>,
          V: VFunction<S>,
          P: Policy
{
    pub fn new<T1, T2, T3>(q_func: Q, v_func: V, policy: P,
                           alpha: T1, beta: T2, gamma: T3) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>
    {
        GreedyGQ {
            q_func: q_func,
            v_func: v_func,

            policy: policy,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, Q, V, P> ControlAgent<S, ActionSpace> for GreedyGQ<S, Q, V, P>
    where Q: QFunction<S> + Linear<S>,
          V: VFunction<S> + Linear<S>,
          P: Policy
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn evaluate_policy(&self, p: &mut Policy, s: &S::Repr) -> usize {
        p.sample(self.q_func.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.phi(s);
        let phi_ns = self.q_func.phi(ns);

        let td_error = t.reward +
            self.q_func.evaluate_action_phi(&(self.gamma.value()*&phi_ns - &phi_s), a);
        let td_estimate = self.v_func.evaluate(s);

        let update_q = td_error*&phi_s - self.gamma*td_estimate*phi_ns;
        let update_v = (td_error - td_estimate)*phi_s;

        self.q_func.update_action_phi(&update_q, a, self.alpha.value());
        self.v_func.update_phi(&update_v, self.alpha*self.beta);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();
    }
}
