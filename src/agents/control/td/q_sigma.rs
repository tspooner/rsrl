use agents::{Controller, Predictor};
use domains::Transition;
use fa::QFunction;
use policies::{Greedy, Policy};
use std::collections::VecDeque;
use {Handler, Parameter, Vector};

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
