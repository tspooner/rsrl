use crate::{
    Algorithm, OnlineLearner, Parameter, Shared, make_shared,
    control::Controller,
    domains::Transition,
    fa::{
        Parameterised, Weights, WeightsView, WeightsViewMut,
        StateActionFunction,
        EnumerableStateActionFunction,
    },
    policies::{Greedy, Policy, FinitePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
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

struct Backup<S> {
    n_steps: usize,
    entries: VecDeque<BackupEntry<S>>,
}

impl<S> Backup<S> {
    pub fn new(n_steps: usize) -> Backup<S> {
        Backup { n_steps, entries: VecDeque::new() }
    }

    pub fn len(&self) -> usize { self.entries.len() }

    pub fn pop(&mut self) -> Option<BackupEntry<S>> { self.entries.pop_front() }

    pub fn push(&mut self, entry: BackupEntry<S>) { self.entries.push_back(entry); }

    pub fn clear(&mut self) { self.entries.clear(); }

    pub fn propagate(&self, gamma: f64) -> (f64, f64) {
        let mut g = self.entries[0].q;
        let mut z = 1.0;
        let mut isr = 1.0;

        for k in 0..(self.n_steps - 1) {
            let b1 = &self.entries[k];
            let b2 = &self.entries[k + 1];

            g += z * b1.residual;
            z *= gamma * ((1.0 - b2.sigma) * b2.pi + b2.sigma);
            isr *= 1.0 - b1.sigma + b1.sigma * b1.pi / b1.mu;
        }

        (isr, g)
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
#[derive(Parameterised)]
pub struct QSigma<S, Q, P> {
    #[weights] pub q_func: Q,

    pub policy: P,
    pub target: Greedy<Q>,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub sigma: Parameter,

    backup: Backup<S>,
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

            backup: Backup::new(n_steps),
        }
    }
}

impl<S, Q: EnumerableStateActionFunction<S>, P> QSigma<S, Q, P> {
    fn update_backup(&mut self, entry: BackupEntry<S>) {
        self.backup.push(entry);

        if self.backup.len() >= self.backup.n_steps {
            let (isr, g) = self.backup.propagate(self.gamma.value());

            let anchor = self.backup.pop().unwrap();
            let qsa = self.q_func.evaluate(&anchor.s, &anchor.a);

            self.q_func.update(
                &anchor.s, &anchor.a,
                self.alpha * isr * (g - qsa),
            );
        }
    }
}

impl<S, Q, P: Algorithm> Algorithm for QSigma<S, Q, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
        self.target.handle_terminal();

        self.backup.clear();
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for QSigma<S, Q, P>
where
    S: Clone,
    Q: EnumerableStateActionFunction<S>,
    P: FinitePolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let qa = self.q_func.evaluate(s, &t.action);
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
            let na = self.policy.sample(&mut thread_rng(), ns);
            let nqs = self.q_func.evaluate_all(ns);

            let pi = self.target.probabilities(&ns);
            let mu = self.policy.probability(ns, &na);

            let exp_nqs = nqs.iter().zip(pi.iter()).fold(0.0, |acc, (q, p)| acc + q * p);
            let residual =
                t.reward + self.gamma * (sigma * nqs[na] + (1.0 - sigma) * exp_nqs) - qa;

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
    Q: EnumerableStateActionFunction<S>,
    P: FinitePolicy<S>,
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
    Q: EnumerableStateActionFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_v(&self, s: &S) -> f64 { self.predict_qsa(s, self.target.mpa(s)) }
}

impl<S, Q, P> ActionValuePredictor<S, <Greedy<Q> as Policy<S>>::Action> for QSigma<S, Q, P>
where
    Q: EnumerableStateActionFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_qsa(&self, s: &S, a: <Greedy<Q> as Policy<S>>::Action) -> f64 {
        self.q_func.evaluate(s, &a)
    }
}
