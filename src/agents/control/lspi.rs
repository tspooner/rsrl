use Parameter;
use agents::{Agent, BatchAgent, LinearAgent, Controller};
use agents::memory::Trace;
use domains::Transition;
use fa::{Function, Projector, Linear};
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use utils::dot;
use std::marker::PhantomData;


pub struct LSPILambda<S: Space, M: Projector<S>, P: Policy> {
    trace: Trace,

    a: Array2<f64>,
    b: Array1<f64>,

    trajectory: Vec<(S::Repr, f64, S::Repr)>,

    pub fa_w: Linear<S, M>,
    pub policy: P,

    pub gamma: Parameter,
    pub lambda: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, M: Projector<S>, P: Policy> LSPILambda<S, M, P> {
    pub fn new<T1, T2>(trace: Trace, fa_w: Linear<S, M>, policy: P, gamma: T1, lambda: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
    {
        let n_features = fa_w.projector.size();

        LSPILambda {
            trace: trace,

            a: Array2::zeros((n_features, n_features)),
            b: Array1::zeros((n_features,)),

            trajectory: vec![],

            fa_w: fa_w,
            policy: policy,

            gamma: gamma.into(),
            lambda: lambda.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, M: Projector<S>, P: Policy> Agent<S> for LSPILambda<S, M, P> {
    type Sample = Transition<S, ActionSpace>;

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let entry = (sample.from.state().clone(), sample.reward, sample.to.state().clone());

        self.trajectory.push(entry);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.consolidate();

        self.gamma = self.gamma.step();
        self.lambda = self.lambda.step();
    }
}

impl<S: Space, M: Projector<S>, P: Policy> BatchAgent<S> for LSPILambda<S, M, P> {
    fn consolidate(&mut self) {
        for (s, r, ns) in self.trajectory.drain(0..) {
            let phi_s = self.fa_w.projector.project_expanded(&s);

            self.trace.decay(self.gamma*self.lambda);
            self.trace.update(&phi_s);

            let z = self.trace.get();
            let z = z.view().into_shape((z.len(), 1)).unwrap();

            let nqs: Vec<f64> = self.fa_w.evaluate(&ns);
            let exp_nqs = dot(&nqs, &Greedy.probabilities(nqs.as_slice()));

            let error_vec = phi_s.clone() - self.gamma*exp_nqs;
            let update_mat = z.dot(&error_vec.view().into_shape((1, phi_s.len())).unwrap());

            self.a.zip_mut_with(&update_mat, move |y, &x| *y += x);
            self.b.scaled_add(r, &phi_s);
        }

        self.fa_w.assign(self.a.solve(&self.b).unwrap().into_shape((self.b.dim(), 1)).unwrap());

        self.a.fill(0.0);
        self.b.fill(0.0);
    }
}

impl<S: Space, M: Projector<S>, P: Policy> LinearAgent<S> for LSPILambda<S, M, P> {
    fn weights(&self) -> Array2<f64> {
        self.fa_w.weights.clone()
    }
}

impl<S: Space, M: Projector<S>, P: Policy> Controller<S, ActionSpace> for LSPILambda<S, M, P> {
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.evaluate_policy(&mut Greedy, s)
    }

    fn mu(&mut self, s: &S::Repr) -> usize {
        let qs: Vec<f64> = self.fa_w.evaluate(s);

        self.policy.sample(qs.as_slice())
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize {
        let qs: Vec<f64> = self.fa_w.evaluate(s);

        p.sample(qs.as_slice())
    }
}
