use core::{Algorithm, Controller, Predictor, Shared, Parameter, Vector, Matrix};
use domains::Transition;
use fa::{Parameterised, QFunction};
use policies::{fixed::Greedy, Policy, FinitePolicy};
use std::collections::VecDeque;

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
pub struct QSigma<S, Q: QFunction<S>, P: Policy<S>> {
    pub q_func: Shared<Q>,

    pub policy: Shared<P>,
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
    P: Policy<S>,
{
    pub fn new<T1, T2, T3>(
        q_func: Shared<Q>,
        policy: Shared<P>,
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

            policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),
            sigma: sigma.into(),

            n_steps,

            backup: VecDeque::new(),
        }
    }

    fn consume_backup(&mut self) {
        let mut g = self.backup[0].q;
        let mut z = 1.0;
        let mut rho = 1.0;

        for k in 0..(self.n_steps - 1) {
            let b1 = &self.backup[k];
            let b2 = &self.backup[k + 1];

            g += z * b1.delta;
            z *= self.gamma * ((1.0 - b2.sigma) * b2.pi + b2.sigma);
            rho *= 1.0 - b1.sigma + b1.sigma * b1.pi / b1.mu;
        }

        let qsa = self.q_func.borrow()
            .evaluate_action(&self.backup[0].s, self.backup[0].a);

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

impl<S: Clone, Q, P> Algorithm<S, P::Action> for QSigma<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    fn handle_sample(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let na = self.sample_behaviour(&ns);
        let pi = self.target.probabilities(&ns);

        let qa = self.predict_qsa(&s, t.action);
        let nqs = self.q_func.borrow().evaluate(&ns).unwrap();
        let nqa = nqs[na];
        let exp_nqs = nqs.dot(&pi);

        let sigma = {
            self.sigma = self.sigma.step();
            self.sigma.value()
        };
        let td_error = t.reward + self.gamma * (sigma * nqa + (1.0 - sigma) * exp_nqs) - qa;

        // Update backup sequence:
        self.backup.push_back(BackupEntry {
            s: s.clone(),
            a: t.action,

            q: qa,
            delta: td_error,

            sigma,
            pi: pi[na],
            mu: self.policy.borrow_mut().probability(ns, na),
        });

        // Learn of latest backup sequence if we have `n_steps` entries:
        if self.backup.len() >= self.n_steps {
            self.consume_backup()
        }
    }

    fn handle_terminal(&mut self, t: &Transition<S, P::Action>) {
        {
            let s = t.from.state();

            let qa = self.predict_qsa(&s, t.action);
            let sigma = {
                self.sigma = self.sigma.step();
                self.sigma.value()
            };

            // Update backup sequence:
            self.backup.push_back(BackupEntry {
                s: s.clone(),
                a: t.action,

                q: qa,
                delta: t.reward - qa,

                sigma,
                pi: 0.0,
                mu: 1.0,
            });

            self.consume_backup();
        }

        self.policy.borrow_mut().handle_terminal(t);

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S: Clone, Q, P> Controller<S, P::Action> for QSigma<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S: Clone, Q, P> Predictor<S, P::Action> for QSigma<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        let a = self.sample_target(s);

        self.q_func.borrow().evaluate(s).unwrap()[a]
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}

impl<S, Q, P> Parameterised for QSigma<S, Q, P>
where
    Q: QFunction<S> + Parameterised,
    P: Policy<S, Action = usize>,
{
    fn weights(&self) -> Matrix<f64> {
        self.q_func.borrow().weights()
    }
}
