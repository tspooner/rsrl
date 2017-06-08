use Parameter;
use fa::QFunction;
use utils::dot;
use agents::ControlAgent;
use domains::Transition;
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};

use std::collections::VecDeque;


struct BackupEntry<S: Space> {
    pub s1: S::Repr,
    pub a1: usize,
    pub s2: S::Repr,
    pub a2: usize,

    pub q1: f64,
    pub q2: f64,
    pub delta: f64,

    pub sigma: f64,
    pub pi: f64,
    pub mu: f64,
}


pub struct QSigma<S: Space, Q: QFunction<S>, P: Policy>
{
    q_func: Q,
    policy: P,

    alpha: Parameter,
    gamma: Parameter,
    sigma: Parameter,

    n_steps: usize,
    backup: VecDeque<BackupEntry<S>>,
}

impl<S: Space, Q, P> QSigma<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    pub fn new<T1, T2, T3>(q_func: Q, policy: P,
                           alpha: T1, gamma: T2,
                           sigma: T3, n_steps: usize) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>,
    {
        QSigma {
            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),
            sigma: sigma.into(),

            n_steps: n_steps,
            backup: VecDeque::new(),
        }
    }

    fn consume_backup(&mut self) {
        let td_error = (1..self.n_steps).fold(self.backup[0].delta, |acc, k| {
            // Calculate the cumulative discount factor:
            let gamma = self.gamma.value();
            let df = (1..k).fold(1.0, |acc, i| {
                let b = &self.backup[i];

                acc * gamma*((1.0 - b.sigma)*b.pi + b.sigma)
            });

            acc + self.backup[k].delta*df
        });

        let isr = (1..self.n_steps).fold(1.0, |acc, k| {
            let b = &self.backup[k];

            acc * (b.sigma*b.pi/b.mu + 1.0 - b.sigma)
        });

        self.q_func.update_action(&self.backup[0].s1, self.backup[0].a1,
                                  self.alpha*isr*td_error);

        self.backup.pop_front();
    }
}

impl<S: Space, Q, P> ControlAgent<S, ActionSpace> for QSigma<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn evaluate_policy(&self, p: &mut Policy, s: &S::Repr) -> usize {
        p.sample(self.q_func.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let q = self.q_func.evaluate_action(s, t.action);
        let nqs = self.q_func.evaluate(ns);

        let npi = self.policy.probabilities(nqs.as_slice());
        let exp_nqs = dot(&nqs, &npi);

        let na = self.policy.sample(nqs.as_slice());
        let nq = nqs[na];

        let sigma = { self.sigma = self.sigma.step(); self.sigma.value() };
        let td_error =
            t.reward + self.gamma*(sigma*nq + (1.0-sigma)*exp_nqs) - q;

        // Update backup sequence:
        self.backup.push_back(BackupEntry {
            s1: s.clone(),
            a1: t.action,
            s2: ns.clone(),
            a2: na,

            q1: q,
            q2: nq,
            delta: td_error,

            sigma: sigma,
            pi: npi[na],
            mu: Greedy.probabilities(&nqs)[na],
        });

        // Learn of latest backup sequence if we have `n_steps` entries:
        if self.backup.len() >= self.n_steps {
            self.consume_backup()
        }
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        // TODO: Handle terminal update according to Sutton's pseudocode.
        //       It's likely that this will require a change to the interface :(
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.sigma = self.sigma.step();
    }
}
