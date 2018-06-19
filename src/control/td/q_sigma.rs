use core::{Algorithm, Controller, Predictor, Shared, Parameter, Vector};
use domains::Transition;
use fa::QFunction;
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

impl<S: Clone, Q, P> Algorithm<S, usize> for QSigma<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let na = self.policy.borrow_mut().sample(&ns);
        let pi = self.target.probabilities(&ns);

        let qa = self.q_func.borrow().evaluate_action(s, t.action);
        let nqs = self.q_func.borrow().evaluate(ns).unwrap();
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

            sigma: sigma,
            pi: pi[na],
            mu: self.policy.borrow_mut().probability(ns, na),
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

        self.policy.borrow_mut().handle_terminal(t);
    }
}

impl<S: Clone, Q, P> Controller<S, usize> for QSigma<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    fn pi(&mut self, s: &S) -> usize { self.target.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.policy.borrow_mut().sample(s) }
}

impl<S: Clone, Q, P> Predictor<S, usize> for QSigma<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    fn v(&mut self, s: &S) -> f64 {
        self.q_func.borrow().evaluate(s).unwrap()[self.target.sample(s)]
    }

    fn qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn qsa(&mut self, s: &S, a: usize) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}
