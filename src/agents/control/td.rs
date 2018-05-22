use agents::{Controller, memory::Trace};
use domains::Transition;
use fa::{Approximator, MultiLFA, Projection, Projector, QFunction};
use policies::{fixed::Greedy, Policy, FinitePolicy};
use std::collections::VecDeque;
use std::marker::PhantomData;
use {Shared, Handler, Parameter, Vector};

/// Watkins' Q-learning.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLearning<S, Q: QFunction<S>, P: FinitePolicy<S>> {
    pub q_func: Shared<Q>,

    pub policy: P,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> QLearning<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    pub fn new<T1, T2>(q_func: Shared<Q>, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        QLearning {
            q_func: q_func.clone(),

            policy: policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, Q, P> Handler<Transition<S, usize>> for QLearning<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.borrow().evaluate(s).unwrap();
        let nqs = self.q_func.borrow().evaluate(ns).unwrap();
        let na = self.target.sample(&ns);

        let td_error = t.reward + self.gamma * nqs[na] - qs[t.action];

        self.q_func.borrow_mut().update_action(s, t.action, self.alpha * td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal(t);
    }
}

impl<S, Q, P> Controller<S, usize> for QLearning<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn pi(&mut self, s: &S) -> usize {
        self.target.sample(s)
    }

    fn mu(&mut self, s: &S) -> usize {
        self.policy.sample(s)
    }
}

/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLambda<S, M: Projector<S>, P: FinitePolicy<S>> {
    trace: Trace,

    pub fa_theta: Shared<MultiLFA<S, M>>,

    pub policy: P,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: 'static, M: Projector<S> + 'static, P: FinitePolicy<S>> QLambda<S, M, P> {
    pub fn new<T1, T2>(
        trace: Trace,
        fa_theta: Shared<MultiLFA<S, M>>,
        policy: P,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        QLambda {
            trace: trace,

            fa_theta: fa_theta.clone(),

            policy: policy,
            target: Greedy::new(fa_theta),

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Handler<Transition<S, usize>> for QLambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.borrow().projector.project(s);
        let phi_ns = self.fa_theta.borrow().projector.project(ns);

        let qs = self.fa_theta.borrow().evaluate_phi(&phi_s);
        let nqs = self.fa_theta.borrow().evaluate_phi(&phi_ns);
        let na = self.target.sample(&ns);

        let td_error = t.reward + self.gamma * nqs[na] - qs[t.action];

        if t.action == self.target.sample(&s) {
            let rate = self.trace.lambda.value() * self.gamma.value();
            self.trace.decay(rate);
        } else {
            self.trace.decay(0.0);
        }

        self.trace
            .update(&phi_s.expanded(self.fa_theta.borrow().projector.dim()));
        self.fa_theta.borrow_mut().update_action_phi(
            &Projection::Dense(self.trace.get()),
            t.action,
            td_error * self.alpha,
        );
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
        self.policy.handle_terminal(t);
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Controller<S, usize> for QLambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize {
        self.target.sample(s)
    }

    fn mu(&mut self, s: &S) -> usize { self.policy.sample(s) }
}

/// On-policy variant of Watkins' Q-learning (aka "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
pub struct SARSA<S, Q: QFunction<S>, P: FinitePolicy<S>> {
    pub q_func: Shared<Q>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> SARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    pub fn new<T1, T2>(q_func: Shared<Q>, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
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

impl<S, Q, P> Handler<Transition<S, usize>> for SARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.borrow().evaluate(s).unwrap();
        let na = self.policy.sample(ns);
        let nq = self.q_func.borrow().evaluate_action(ns, na);

        let td_error = t.reward + self.gamma * nq - qs[t.action];

        self.q_func.borrow_mut().update_action(s, t.action, self.alpha * td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal(t);
    }
}

impl<S, Q, P> Controller<S, usize> for SARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn pi(&mut self, s: &S) -> usize {self.policy.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }
}

/// On-policy variant of Watkins' Q-learning with eligibility traces (aka
/// "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
pub struct SARSALambda<S, M: Projector<S>, P: FinitePolicy<S>> {
    trace: Trace,

    pub fa_theta: Shared<MultiLFA<S, M>>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> SARSALambda<S, M, P> {
    pub fn new<T1, T2>(
        trace: Trace,
        fa_theta: Shared<MultiLFA<S, M>>,
        policy: P,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        SARSALambda {
            trace: trace,

            fa_theta: fa_theta,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Handler<Transition<S, usize>> for SARSALambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.borrow().projector.project(s);

        let qsa = self.fa_theta.borrow().evaluate_action_phi(&phi_s, t.action);
        let na = self.policy.sample(ns);
        let nq = self.fa_theta.borrow().evaluate_action(ns, na);

        let rate = self.trace.lambda.value() * self.gamma.value();
        let td_error = t.reward + self.gamma * nq - qsa;

        self.trace.decay(rate);
        self.trace
            .update(&phi_s.expanded(self.fa_theta.borrow().projector.dim()));

        self.fa_theta.borrow_mut().update_action_phi(
            &Projection::Dense(self.trace.get()),
            t.action,
            self.alpha * td_error,
        );
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
        self.policy.handle_terminal(t);
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Controller<S, usize> for SARSALambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize { self.policy.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }
}

/// Action probability-weighted variant of SARSA (aka "summation Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - van Seijen, H., van Hasselt, H., Whiteson, S., Wiering, M. (2009). A
/// theoretical and empirical analysis of Expected Sarsa. In Proceedings of the
/// IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning,
/// pp. 177–184.
pub struct ExpectedSARSA<S, Q: QFunction<S>, P: FinitePolicy<S>> {
    pub q_func: Shared<Q>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> ExpectedSARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    pub fn new<T1, T2>(q_func: Shared<Q>, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
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

impl<S, Q, P> Handler<Transition<S, usize>> for ExpectedSARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.borrow().evaluate(s).unwrap();
        let nqs = self.q_func.borrow().evaluate(ns).unwrap();

        let exp_nqs = self.policy.probabilities(s).dot(&nqs);
        let td_error = t.reward + self.gamma * exp_nqs - qs[t.action];

        self.q_func.borrow_mut().update_action(s, t.action, self.alpha * td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal(t);
    }
}

impl<S, Q, P> Controller<S, usize> for ExpectedSARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn pi(&mut self, s: &S) -> usize { self.policy.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }
}

/// General multi-step temporal-difference learning algorithm.
///
/// # Parameters
/// - `sigma` varies the degree of sampling, yielding classical learning
/// algorithms as special cases:
///     * `0` - `ExpectedSARSA` | `TreeBackup`
///     * `1` - `SARSA`
///
/// # References
/// - Sutton, R. S. and Barto, A. G. (2017). Reinforcement Learning: An
/// Introduction (2nd ed.). Manuscript in preparation.
/// - De Asis, K., Hernandez-Garcia, J. F., Holland, G. Z., & Sutton, R. S.
/// (2017). Multi-step Reinforcement Learning: A Unifying Algorithm. arXiv
/// preprint arXiv:1703.01327.
pub struct QSigma<S, Q: QFunction<S>, P: FinitePolicy<S>> {
    pub q_func: Shared<Q>,

    pub policy: P,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub sigma: Parameter,

    pub n_steps: usize,

    backup: VecDeque<BackupEntry<S>>,
}

impl<S, Q, P> QSigma<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    pub fn new<T1, T2, T3>(
        q_func: Shared<Q>,
        policy: P,
        alpha: T1,
        gamma: T2,
        sigma: T3,
        n_steps: usize,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        QSigma {
            q_func: q_func.clone(),

            policy: policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),
            sigma: sigma.into(),

            n_steps: n_steps,

            backup: VecDeque::new(),
        }
    }

    fn consume_backup(&mut self) {
        let mut g = self.backup[0].q;
        let mut z = 1.0;
        let mut rho = 1.0;

        for k in 0..(self.n_steps - 1) {
            let ref b1 = self.backup[k];
            let ref b2 = self.backup[k + 1];

            g += z * b1.delta;
            z *= self.gamma * ((1.0 - b2.sigma) * b2.pi + b2.sigma);
            rho *= 1.0 - b1.sigma + b1.sigma * b1.pi / b1.mu;
        }

        let qsa = self.q_func.borrow().evaluate_action(&self.backup[0].s, self.backup[0].a);

        self.q_func.borrow_mut().update_action(
            &self.backup[0].s,
            self.backup[0].a,
            self.alpha * rho * (g - qsa),
        );

        self.backup.pop_front();
    }
}

struct BackupEntry<S> {
    pub s: S,
    pub a: usize,

    pub q: f64,
    pub delta: f64,

    pub sigma: f64,
    pub pi: f64,
    pub mu: f64,
}

impl<S: Clone, Q, P> Handler<Transition<S, usize>> for QSigma<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let na = self.policy.sample(ns);
        let nmu = self.policy.probability(ns, na);
        let npi = self.target.probabilities(&ns);

        let q = self.q_func.borrow().evaluate_action(s, t.action);
        let nqs = self.q_func.borrow().evaluate(ns).unwrap();
        let exp_nqs = nqs.dot(&npi);

        let sigma = {
            self.sigma = self.sigma.step();
            self.sigma.value()
        };
        let td_error = t.reward + self.gamma * (sigma * nqs[na] + (1.0 - sigma) * exp_nqs) - q;

        // Update backup sequence:
        self.backup.push_back(BackupEntry {
            s: s.clone(),
            a: t.action,

            q: q,
            delta: td_error,

            sigma: sigma,
            pi: npi[na],
            mu: nmu,
        });

        // Learn of latest backup sequence if we have `n_steps` entries:
        if self.backup.len() >= self.n_steps {
            self.consume_backup()
        }
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        // TODO: Handle terminal update according to Sutton's pseudocode.
        //       It's likely that this will require a change to the interface
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal(t);
    }
}

impl<S: Clone, Q, P> Controller<S, usize> for QSigma<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    fn pi(&mut self, s: &S) -> usize {
        self.target.sample(&s)
    }

    fn mu(&mut self, s: &S) -> usize {
        self.policy.sample(s)
    }
}

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
        let na = self.target.sample(&ns);

        let vs = qs[self.target.sample(&s)];

        let td_error = t.reward + self.gamma * nqs[na] - qs[t.action];
        let al_error = td_error - self.alpha * (vs - qs[t.action]);
        let pal_error = al_error.max(td_error - self.alpha * (vs - nqs[t.action]));

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

// TODO:
// PQ(lambda) - http://proceedings.mlr.press/v32/sutton14.pdf
