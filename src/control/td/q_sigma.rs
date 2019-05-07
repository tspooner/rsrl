use crate::{
    core::*,
    domains::Transition,
    fa::{Parameterised, Features, QFunction},
    geometry::{MatrixView, MatrixViewMut},
    policies::{Greedy, Policy, FinitePolicy},
};
use rand::{thread_rng, Rng};
use std::collections::VecDeque;

struct BackupEntry<S> {
    pub s: S,
    pub a: usize,

    pub q: f64,
    pub residual: f64,

    pub sigma: f64,
    pub pi: f64,
    pub mu: f64,
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
#[derive(Parameterised)]
pub struct QSigma<S, Q, P> {
    #[weights] pub q_func: Q,

    pub policy: P,
    pub target: Greedy<Q>,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub sigma: Parameter,
    pub n_steps: usize,

    backup: VecDeque<BackupEntry<S>>,
}

impl<S, Q, P> QSigma<S, Shared<Q>, P> {
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
        let q_func = make_shared(q_func);

        QSigma {
            q_func: q_func.clone(),

            policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),
            sigma: sigma.into(),
            n_steps,

            backup: VecDeque::new(),
        }
    }
}

impl<S, Q: QFunction<S>, P> QSigma<S, Q, P> {
    fn consume_backup(&mut self) {
        let mut g = self.backup[0].q;
        let mut z = 1.0;
        let mut rho = 1.0;

        for k in 0..(self.n_steps - 1) {
            let b1 = &self.backup[k];
            let b2 = &self.backup[k + 1];

            g += z * b1.residual;
            z *= self.gamma * ((1.0 - b2.sigma) * b2.pi + b2.sigma);
            rho *= 1.0 - b1.sigma + b1.sigma * b1.pi / b1.mu;
        }

        let phi_s = self.q_func.embed(&self.backup[0].s);
        let qsa = self.q_func.evaluate_index(&phi_s, self.backup[0].a).unwrap();

        self.q_func.update_index(
            &phi_s,
            self.backup[0].a,
            self.alpha * rho * (g - qsa),
        ).ok();

        self.backup.pop_front();
    }

    #[inline(always)]
    fn update_backup(&mut self, entry: BackupEntry<S>) {
        self.backup.push_back(entry);

        if self.backup.len() >= self.n_steps {
            self.consume_backup()
        }
    }
}

impl<S, Q, P: Algorithm> Algorithm for QSigma<S, Q, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
        self.target.handle_terminal();
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for QSigma<S, Q, P>
where
    S: Clone,
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let phi_s = self.q_func.embed(s);
        let qa = self.q_func.evaluate_index(&phi_s, t.action).unwrap();
        let sigma = {
            self.sigma = self.sigma.step();
            self.sigma.value()
        };

        if t.terminated() {
            self.update_backup(BackupEntry {
                s: s.clone(),
                a: t.action,

                q: qa,
                residual: t.reward - qa,

                sigma: sigma,
                pi: 0.0,
                mu: 1.0,
            });

            self.backup.clear();

        } else {
            let ns = t.to.state();
            let na = self.sample_behaviour(&mut thread_rng(), ns);

            let phi_ns = self.q_func.embed(ns);
            let nqs = self.q_func.evaluate(&phi_ns).unwrap();
            let nqa = nqs[na];

            let pi = self.target.probabilities(&ns);
            let exp_nqs = nqs.dot(&pi);

            let mu = self.policy.probability(ns, &na);

            let residual = t.reward + self.gamma * (sigma * nqa + (1.0 - sigma) * exp_nqs) - qa;

            self.update_backup(BackupEntry {
                s: s.clone(),
                a: t.action,

                q: qa,
                residual: residual,

                sigma: sigma,
                pi: pi[na],
                mu: mu,
            });
        };
    }
}

impl<S, Q, P> Controller<S, P::Action> for QSigma<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.target.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}

impl<S, Q, P> ValuePredictor<S> for QSigma<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn predict_v(&self, s: &S) -> f64 { self.predict_qsa(s, self.target.mpa(s)) }
}

impl<S, Q, P> ActionValuePredictor<S, P::Action> for QSigma<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn predict_qs(&self, s: &S) -> Vector<f64> {
        self.q_func.evaluate(&self.q_func.embed(s)).unwrap()
    }

    fn predict_qsa(&self, s: &S, a: P::Action) -> f64 {
        self.q_func.evaluate_index(&self.q_func.embed(s), a).unwrap()
    }
}
