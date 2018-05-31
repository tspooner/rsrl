use agents::{Controller, Predictor, memory::Trace};
use domains::Transition;
use fa::{Approximator, MultiLFA, Projection, Projector, QFunction};
use policies::{Greedy, Policy};
use std::{collections::VecDeque, marker::PhantomData};
use {Handler, Parameter, Vector};

/// Watkins' Q-learning.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLearning<S, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> QLearning<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
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

impl<S, Q, P> Handler<Transition<S, usize>> for QLearning<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s).unwrap();
        let nqs = self.q_func.evaluate(ns).unwrap();

        let a = t.action;
        let na = Greedy.sample(nqs.as_slice().unwrap());

        let td_error = t.reward + self.gamma * nqs[na] - qs[a];

        self.q_func.update_action(s, a, self.alpha * td_error);
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S, Q, P> Controller<S, usize> for QLearning<S, Q, P>
where
    Q: QFunction<S>,
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

impl<S, Q: QFunction<S>, P: Policy> Predictor<S> for QLearning<S, Q, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.q_func.evaluate(s).unwrap();
        let na = Greedy.sample(nqs.as_slice().unwrap());

        nqs[na]
    }
}

/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLambda<S, M: Projector<S>, P: Policy> {
    trace: Trace,

    pub fa_theta: MultiLFA<S, M>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M: Projector<S>, P: Policy> QLambda<S, M, P> {
    pub fn new<T1, T2>(
        trace: Trace,
        fa_theta: MultiLFA<S, M>,
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

            fa_theta: fa_theta,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: Policy> Handler<Transition<S, usize>> for QLambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.projector.project(s);
        let phi_ns = self.fa_theta.projector.project(ns);

        let qs = self.fa_theta.evaluate_phi(&phi_s);
        let nqs = self.fa_theta.evaluate_phi(&phi_ns);
        let na = Greedy.sample(nqs.as_slice().unwrap());

        let td_error = t.reward + self.gamma * nqs[na] - qs[a];

        if a == Greedy.sample(qs.as_slice().unwrap()) {
            let rate = self.trace.lambda.value() * self.gamma.value();
            self.trace.decay(rate);
        } else {
            self.trace.decay(0.0);
        }

        self.trace
            .update(&phi_s.expanded(self.fa_theta.projector.dim()));
        self.fa_theta.update_action_phi(
            &Projection::Dense(self.trace.get()),
            a,
            td_error * self.alpha,
        );
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
        self.policy.handle_terminal();
    }
}

impl<S, M: Projector<S>, P: Policy> Controller<S, usize> for QLambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        Greedy.sample(qs.as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        self.policy.sample(qs.as_slice().unwrap())
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        p.sample(qs.as_slice().unwrap())
    }
}

impl<S, M: Projector<S>, P: Policy> Predictor<S> for QLambda<S, M, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.fa_theta.evaluate(s).unwrap();
        let na = Greedy.sample(nqs.as_slice().unwrap());

        nqs[na]
    }
}

/// On-policy variant of Watkins' Q-learning (aka "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
pub struct SARSA<S, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> SARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
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
    P: Policy,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s).unwrap();
        let nqs = self.q_func.evaluate(ns).unwrap();

        let a = t.action;
        let na = self.policy.sample(nqs.as_slice().unwrap());

        let td_error = t.reward + self.gamma * nqs[na] - qs[a];

        self.q_func.update_action(s, a, self.alpha * td_error);
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S, Q, P> Controller<S, usize> for SARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    fn pi(&mut self, s: &S) -> usize {
        self.policy
            .sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        p.sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }
}

impl<S, Q: QFunction<S>, P: Policy> Predictor<S> for SARSA<S, Q, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.q_func.evaluate(s).unwrap();
        let pi: Vector<f64> = self.policy.probabilities(nqs.as_slice().unwrap()).into();

        pi.dot(&nqs)
    }
}

/// On-policy variant of Watkins' Q-learning with eligibility traces (aka
/// "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
pub struct SARSALambda<S, M: Projector<S>, P: Policy> {
    trace: Trace,

    pub fa_theta: MultiLFA<S, M>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M: Projector<S>, P: Policy> SARSALambda<S, M, P> {
    pub fn new<T1, T2>(
        trace: Trace,
        fa_theta: MultiLFA<S, M>,
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

impl<S, M: Projector<S>, P: Policy> Handler<Transition<S, usize>> for SARSALambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.projector.project(s);
        let phi_ns = self.fa_theta.projector.project(ns);

        let qsa = self.fa_theta.evaluate_action_phi(&phi_s, a);
        let nqs = self.fa_theta.evaluate_phi(&phi_ns);
        let na = self.policy.sample(nqs.as_slice().unwrap());

        let rate = self.trace.lambda.value() * self.gamma.value();
        let td_error = t.reward + self.gamma * nqs[na] - qsa;

        self.trace.decay(rate);
        self.trace
            .update(&phi_s.expanded(self.fa_theta.projector.dim()));

        self.fa_theta.update_action_phi(
            &Projection::Dense(self.trace.get()),
            a,
            self.alpha * td_error,
        );
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
        self.policy.handle_terminal();
    }
}

impl<S, M: Projector<S>, P: Policy> Controller<S, usize> for SARSALambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        self.policy.sample(qs.as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        p.sample(qs.as_slice().unwrap())
    }
}

impl<S, M: Projector<S>, P: Policy> Predictor<S> for SARSALambda<S, M, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.fa_theta.evaluate(s).unwrap();
        let pi: Vector<f64> = self.policy.probabilities(nqs.as_slice().unwrap()).into();

        pi.dot(&nqs)
    }
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
pub struct ExpectedSARSA<S, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> ExpectedSARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
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
    P: Policy,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s).unwrap();
        let nqs = self.q_func.evaluate(ns).unwrap();

        let a = t.action;

        let pi: Vector<f64> = self.policy.probabilities(nqs.as_slice().unwrap()).into();
        let exp_nqs = pi.dot(&nqs);
        let td_error = t.reward + self.gamma * exp_nqs - qs[a];

        self.q_func.update_action(s, a, self.alpha * td_error);
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S, Q, P> Controller<S, usize> for ExpectedSARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    fn pi(&mut self, s: &S) -> usize {
        self.policy
            .sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        p.sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }
}

impl<S, Q: QFunction<S>, P: Policy> Predictor<S> for ExpectedSARSA<S, Q, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.q_func.evaluate(s).unwrap();
        let pi: Vector<f64> = self.policy.probabilities(nqs.as_slice().unwrap()).into();

        pi.dot(&nqs)
    }
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
pub struct QSigma<S, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub sigma: Parameter,

    pub n_steps: usize,

    backup: VecDeque<BackupEntry<S>>,
}

impl<S, Q, P> QSigma<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    pub fn new<T1, T2, T3>(
        q_func: Q,
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

        let qsa = self.q_func
            .evaluate_action(&self.backup[0].s, self.backup[0].a);
        self.q_func.update_action(
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
    Q: QFunction<S>,
    P: Policy,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let nqs = self.q_func.evaluate(ns).unwrap();
        let nqs_slice = nqs.as_slice().unwrap();

        let na = self.policy.sample(nqs_slice);
        let nmu = self.policy.probabilities(nqs_slice);
        let npi: Vector<f64> = Greedy.probabilities(nqs_slice).into();

        let q = self.q_func.evaluate_action(s, t.action);
        let nq = nqs[na];
        let exp_nqs = nqs.dot(&npi);

        let sigma = {
            self.sigma = self.sigma.step();
            self.sigma.value()
        };
        let td_error = t.reward + self.gamma * (sigma * nq + (1.0 - sigma) * exp_nqs) - q;

        // Update backup sequence:
        self.backup.push_back(BackupEntry {
            s: s.clone(),
            a: t.action,

            q: q,
            delta: td_error,

            sigma: sigma,
            pi: npi[na],
            mu: nmu[na],
        });

        // Learn of latest backup sequence if we have `n_steps` entries:
        if self.backup.len() >= self.n_steps {
            self.consume_backup()
        }
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        // TODO: Handle terminal update according to Sutton's pseudocode.
        //       It's likely that this will require a change to the interface
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S: Clone, Q, P> Controller<S, usize> for QSigma<S, Q, P>
where
    Q: QFunction<S>,
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

impl<S, Q: QFunction<S>, P: Policy> Predictor<S> for QSigma<S, Q, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.q_func.evaluate(s).unwrap();
        let na = Greedy.sample(nqs.as_slice().unwrap());

        nqs[na]
    }
}

/// Persistent Advantage Learning
///
/// # References
/// - Bellemare, Marc G., et al. "Increasing the Action Gap: New Operators for
/// Reinforcement Learning." AAAI. 2016.
pub struct PAL<S, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        PAL {
            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, Q, P> Handler<Transition<S, usize>> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s).unwrap();
        let nqs = self.q_func.evaluate(ns).unwrap();

        let a = t.action;
        let na = Greedy.sample(nqs.as_slice().unwrap());

        let vs = qs[Greedy.sample(qs.as_slice().unwrap())];

        let td_error = t.reward + self.gamma * nqs[na] - qs[a];
        let al_error = td_error - self.alpha * (vs - qs[a]);
        let pal_error = al_error.max(td_error - self.alpha * (vs - nqs[a]));

        self.q_func.update_action(s, a, self.alpha * pal_error);
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S, Q, P> Controller<S, usize> for PAL<S, Q, P>
where
    Q: QFunction<S>,
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

impl<S, Q: QFunction<S>, P: Policy> Predictor<S> for PAL<S, Q, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.q_func.evaluate(s).unwrap();
        let na = Greedy.sample(nqs.as_slice().unwrap());

        nqs[na]
    }
}

// TODO:
// PQ(lambda) - http://proceedings.mlr.press/v32/sutton14.pdf
