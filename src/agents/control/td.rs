use Parameter;
use agents::ControlAgent;
use agents::memory::Trace;
use domains::Transition;
use fa::{Function, QFunction, Projector, Projection, Linear};
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};
use std::collections::VecDeque;
use std::marker::PhantomData;
use utils::dot;


/// Watkins' Q-learning.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis, Cambridge
/// University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning, 8:279–292.
pub struct QLearning<S: Space, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, Q, P> QLearning<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
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
        Greedy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn mu(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize {
        p.sample(self.q_func.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s);
        let nqs = self.q_func.evaluate(ns);

        let a = t.action;
        let na = Greedy.sample(nqs.as_slice());

        let td_error = t.reward + self.gamma*nqs[na] - qs[a];

        self.q_func.update_action(s, a, self.alpha*td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}


/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis, Cambridge
/// University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning, 8:279–292.
pub struct QLambda<S: Space, M: Projector<S>, P: Policy> {
    trace: Trace,

    pub q_func: Linear<S, M>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, M, P> QLambda<S, M, P>
    where M: Projector<S>,
          P: Policy
{
    pub fn new<T1, T2>(trace: Trace, q_func: Linear<S, M>, policy: P, alpha: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        QLambda {
            trace: trace,

            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, M, P> ControlAgent<S, ActionSpace> for QLambda<S, M, P>
    where M: Projector<S>,
          P: Policy
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        let qs: Vec<f64> = self.q_func.evaluate(s);

        Greedy.sample(&qs)
    }

    fn mu(&mut self, s: &S::Repr) -> usize {
        let qs: Vec<f64> = self.q_func.evaluate(s);

        self.policy.sample(&qs)
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize {
        let qs: Vec<f64> = self.q_func.evaluate(s);

        p.sample(&qs)
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.projector.project(s);
        let phi_ns = self.q_func.projector.project(ns);

        let nqs = self.q_func.evaluate_phi(&phi_ns);
        let na = Greedy.sample(nqs.as_slice());

        let td_error = t.reward + self.gamma*nqs[na] - self.q_func.evaluate_action_phi(&phi_s, a);

        self.trace.decay(self.gamma.value());
        self.trace.update(&self.q_func.projector.expand_projection(phi_s));

        self.q_func.update_action_phi(&Projection::Dense(self.trace.get()), a, self.alpha*td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}


/// On-policy variant of Watkins' Q-learning (aka "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D thesis, Cambridge
/// University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing eligibility traces.
/// Machine Learning 22:123–158.
pub struct SARSA<S: Space, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, Q, P> SARSA<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
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

    fn mu(&mut self, s: &S::Repr) -> usize {
        self.pi(s)
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize {
        p.sample(self.q_func.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s);
        let nqs = self.q_func.evaluate(ns);

        let a = t.action;
        let na = self.policy.sample(nqs.as_slice());

        let td_error = t.reward + self.gamma*nqs[na] - qs[a];

        self.q_func.update_action(s, a, self.alpha*td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}


/// On-policy variant of Watkins' Q-learning with eligibility traces (aka "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D thesis, Cambridge
/// University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing eligibility traces.
/// Machine Learning 22:123–158.
pub struct SARSALambda<S: Space, M: Projector<S>, P: Policy> {
    trace: Trace,

    pub q_func: Linear<S, M>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, M, P> SARSALambda<S, M, P>
    where M: Projector<S>,
          P: Policy
{
    pub fn new<T1, T2>(trace: Trace, q_func: Linear<S, M>, policy: P, alpha: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        SARSALambda {
            trace: trace,

            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, M, P> ControlAgent<S, ActionSpace> for SARSALambda<S, M, P>
    where M: Projector<S>,
          P: Policy
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        let qs: Vec<f64> = self.q_func.evaluate(s);

        self.policy.sample(&qs)
    }

    fn mu(&mut self, s: &S::Repr) -> usize {
        self.pi(s)
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize {
        let qs: Vec<f64> = self.q_func.evaluate(s);

        p.sample(&qs)
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.projector.project(s);
        let phi_ns = self.q_func.projector.project(ns);

        let nqs = self.q_func.evaluate_phi(&phi_ns);
        let na = self.policy.sample(nqs.as_slice());

        let td_error = t.reward + self.gamma*nqs[na] - self.q_func.evaluate_action_phi(&phi_s, a);

        self.trace.decay(self.gamma.value());
        self.trace.update(&self.q_func.projector.expand_projection(phi_s));

        self.q_func.update_action_phi(&Projection::Dense(self.trace.get()), a, self.alpha*td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}


/// Action probability-weighted variant of SARSA (aka "summation Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D thesis, Cambridge
/// University.
/// - van Seijen, H., van Hasselt, H., Whiteson, S., Wiering, M. (2009). A theoretical and
/// empirical analysis of Expected Sarsa. In Proceedings of the IEEE Symposium on Adaptive Dynamic
/// Programming and Reinforcement Learning, pp. 177–184.
pub struct ExpectedSARSA<S: Space, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, Q, P> ExpectedSARSA<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
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

    fn mu(&mut self, s: &S::Repr) -> usize {
        self.pi(s)
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize {
        p.sample(self.q_func.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s);
        let nqs = self.q_func.evaluate(ns);

        let a = t.action;

        let exp_nqs = dot(&nqs, &self.policy.probabilities(nqs.as_slice()));
        let td_error = t.reward + self.gamma*exp_nqs - qs[a];

        self.q_func.update_action(s, a, self.alpha*td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}


/// General multi-step temporal-difference learning algorithm.
///
/// # Parameters
/// - `sigma` varies the degree of sampling, yielding classical learning algorithms as special
/// cases:
///     * `0` - `ExpectedSARSA` | `TreeBackup`
///     * `1` - `SARSA`
///
/// # References
/// - Sutton, R. S. and Barto, A. G. (2017). Reinforcement Learning: An Introduction (2nd ed.).
/// Manuscript in preparation.
/// - De Asis, K., Hernandez-Garcia, J. F., Holland, G. Z., & Sutton, R. S. (2017). Multi-step
/// Reinforcement Learning: A Unifying Algorithm. arXiv preprint arXiv:1703.01327.
pub struct QSigma<S: Space, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub sigma: Parameter,

    pub n_steps: usize,

    backup: VecDeque<BackupEntry<S>>,
}

impl<S: Space, Q, P> QSigma<S, Q, P>
    where Q: QFunction<S>,
          P: Policy
{
    pub fn new<T1, T2, T3>(q_func: Q,
                           policy: P,
                           alpha: T1,
                           gamma: T2,
                           sigma: T3,
                           n_steps: usize)
                           -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>
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

                acc*gamma*((1.0 - b.sigma)*b.pi + b.sigma)
            });

            acc + self.backup[k].delta*df
        });

        let isr = (1..self.n_steps).fold(1.0, |acc, k| {
            let b = &self.backup[k];

            acc*(b.sigma*b.mu / b.pi + 1.0 - b.sigma)
        });

        self.q_func.update_action(&self.backup[0].s1,
                                  self.backup[0].a1,
                                  self.alpha*isr*td_error);

        self.backup.pop_front();
    }
}

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

impl<S: Space, Q, P> ControlAgent<S, ActionSpace> for QSigma<S, Q, P>
    where Q: QFunction<S>,
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

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let q = self.q_func.evaluate_action(s, t.action);
        let nqs = self.q_func.evaluate(ns);

        let npi = self.policy.probabilities(nqs.as_slice());
        let exp_nqs = dot(&nqs, &npi);

        let na = self.policy.sample(nqs.as_slice());
        let nq = nqs[na];

        let sigma = {
            self.sigma = self.sigma.step();
            self.sigma.value()
        };
        let td_error = t.reward + self.gamma*(sigma*nq + (1.0 - sigma)*exp_nqs) - q;

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
        //       It's likely that this will require a change to the interface
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

// TODO:
// PQ(lambda) - http://proceedings.mlr.press/v32/sutton14.pdf
