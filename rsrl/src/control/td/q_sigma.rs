use crate::{
    domains::Transition,
    fa::StateActionUpdate,
    policies::EnumerablePolicy,
    prediction::{ActionValuePredictor, ValuePredictor},
    utils::argmaxima,
    Enumerable,
    Function,
    Handler,
    Parameterised,
};
use rand::thread_rng;
use std::{collections::VecDeque, ops::Index};

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
        Backup {
            n_steps,
            entries: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize { self.entries.len() }

    pub fn pop(&mut self) -> Option<BackupEntry<S>> { self.entries.pop_front() }

    pub fn push(&mut self, entry: BackupEntry<S>) { self.entries.push_back(entry); }

    pub fn clear(&mut self) { self.entries.clear(); }

    pub fn propagate(&self, gamma: f64) -> (f64, f64) {
        let mut g = self.entries[0].q;
        let mut z = 1.0;
        let mut isr = 1.0;

        for k in 0..self.n_steps {
            let b1 = &self.entries[k];
            let b2 = &self.entries[k + 1];

            g += z * b1.residual;
            z *= gamma * ((1.0 - b1.sigma) * b2.pi + b2.sigma);
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
    #[weights]
    pub q_func: Q,
    pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
    pub sigma: f64,

    backup: Backup<S>,
}

impl<S, Q, P> QSigma<S, Q, P> {
    pub fn new(q_func: Q, policy: P, alpha: f64, gamma: f64, sigma: f64, n_steps: usize) -> Self {
        QSigma {
            q_func,
            policy,

            alpha,
            gamma,
            sigma,

            backup: Backup::new(n_steps),
        }
    }
}

impl<S, Q, P> QSigma<S, Q, P> {
    fn update_backup(&mut self, entry: BackupEntry<S>) -> Option<Result<Q::Response, Q::Error>>
    where
        Q: for<'s, 'a> Function<(&'s S, &'a usize), Output = f64>,
        Q: Handler<StateActionUpdate<S, usize, f64>>,
    {
        self.backup.push(entry);

        if self.backup.len() >= self.backup.n_steps {
            let (isr, g) = self.backup.propagate(self.gamma);

            let anchor = self.backup.pop().unwrap();
            let qsa = self.q_func.evaluate((&anchor.s, &anchor.a));

            Some(self.q_func.handle(StateActionUpdate {
                state: anchor.s,
                action: anchor.a,
                error: self.alpha * isr * (g - qsa),
            }))
        } else {
            None
        }
    }
}

impl<'m, S, Q, P> Handler<&'m Transition<S, usize>> for QSigma<S, Q, P>
where
    S: Clone,

    Q: Enumerable<(&'m S,)>
        + for<'s, 'a> Function<(&'s S, &'a usize), Output = f64>
        + Handler<StateActionUpdate<S, usize, f64>>,
    P: EnumerablePolicy<&'m S>,

    <Q as Function<(&'m S,)>>::Output: Index<usize, Output = f64> + IntoIterator<Item = f64>,
    <<Q as Function<(&'m S,)>>::Output as IntoIterator>::IntoIter: ExactSizeIterator,

    <P as Function<(&'m S,)>>::Output: Index<usize, Output = f64> + IntoIterator<Item = f64>,
    <<P as Function<(&'m S,)>>::Output as IntoIterator>::IntoIter: ExactSizeIterator,
{
    type Response = Option<Q::Response>;
    type Error = Q::Error;

    fn handle(&mut self, t: &'m Transition<S, usize>) -> Result<Self::Response, Self::Error> {
        let s = t.from.state();
        let qa = self.q_func.evaluate_index((s,), t.action);

        let res = if t.terminated() {
            let res = self.update_backup(BackupEntry {
                s: s.clone(),
                a: t.action,

                q: qa,
                residual: t.reward - qa,

                sigma: self.sigma,
                pi: 0.0,
                mu: 1.0,
            });
            self.backup.clear();

            res
        } else {
            let ns = t.to.state();
            let na = self.policy.sample(&mut thread_rng(), ns);
            let nqs = self.q_func.evaluate((ns,));
            let nqsna = nqs[na];

            let (na_max, exp_nqs) = argmaxima(nqs.into_iter());

            let pi = if na_max.contains(&na) {
                1.0 / na_max.len() as f64
            } else {
                0.0
            };
            let mu = self.policy.evaluate((ns, na));

            let residual =
                t.reward + self.gamma * (self.sigma * nqsna + (1.0 - self.sigma) * exp_nqs) - qa;

            self.update_backup(BackupEntry {
                s: s.clone(),
                a: t.action,

                q: qa,
                residual: residual,

                sigma: self.sigma,
                pi: pi,
                mu: mu,
            })
        };

        res.transpose()
    }
}

impl<S, Q, P> ValuePredictor<S> for QSigma<S, Q, P>
where
    Q: Enumerable<(S,)>,
    <Q as Function<(S,)>>::Output: Index<usize, Output = f64> + IntoIterator<Item = f64>,
    <<Q as Function<(S,)>>::Output as IntoIterator>::IntoIter: ExactSizeIterator,
{
    fn predict_v(&self, s: S) -> f64 { self.q_func.find_max((s,)).1 }
}

impl<S, Q, P> ActionValuePredictor<S, usize> for QSigma<S, Q, P>
where Q: Function<(S, usize), Output = f64>
{
    fn predict_q(&self, s: S, a: usize) -> f64 { self.q_func.evaluate((s, a)) }
}
