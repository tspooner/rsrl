use crate::{
    core::DerefSlice,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction, StateActionFunction, DifferentiableStateActionFunction,
        EnumerableStateActionFunction,
        linear::{
            LinearStateFunction, LinearStateActionFunction,
            Approximator, ScalarFunction, LFAGradient, Features,
            basis::Projector,
            optim::Optimiser,
        }
    },
    policies::{Policy, DifferentiablePolicy},
};
use ndarray::Axis;

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Parameterised)]
pub struct CFA<P, O> {
    pub policy: P,
    pub optimiser: O,
    #[weights] pub approximator: ScalarFunction,
}

impl<P: Parameterised, O: Optimiser> CFA<P, O> {
    pub fn new(policy: P, optimiser: O) -> Self {
        let wd = policy.weights_dim();
        let approximator = ScalarFunction::zeros(wd[0] * wd[1]);

        CFA { policy, optimiser, approximator }
    }
}

// Q(x, u):
impl<X, U, P, O> StateActionFunction<X, U> for CFA<P, O>
where
    P: DifferentiablePolicy<X, Action = U>,
    O: Optimiser,
{
    type Output = f64;

    fn evaluate(&self, state: &X, action: &U) -> Self::Output {
        let features = self.features(state, action);

        self.approximator.evaluate(&features).unwrap()
    }

    fn update(&mut self, state: &X, action: &U, error: Self::Output) {
        let features = self.features(state, action);

        self.approximator.update(&mut self.optimiser, &features, error).ok();
    }
}

impl<X, U, P, O> DifferentiableStateActionFunction<X, U> for CFA<P, O>
where
    P: DifferentiablePolicy<X, Action = U>,
    O: Optimiser,
{
    type Gradient = LFAGradient;

    fn grad(&self, state: &X, action: &U) -> Self::Gradient {
        LFAGradient::from_features(
            [self.n_features(), 1], 0,
            self.features(state, action)
        )
    }

    // fn zero_grad(&self) -> Self::Gradient {
        // LFAGradient::empty(self.weights_dim())
    // }
}

impl<X, U, P, O> LinearStateActionFunction<X, U> for CFA<P, O>
where
    P: DifferentiablePolicy<X, Action = U>,
    O: Optimiser,
{
    fn n_features(&self) -> usize {
        let wd = self.policy.weights_dim();

        wd[0] * wd[1]
    }

    fn features(&self, state: &X, action: &U) -> Features {
        let gl_policy = self.policy.grad_log(state, action);

        Features::Dense(gl_policy.index_axis_move(Axis(1), 0) )
    }

    fn evaluate_features(&self, features: &Features, _: &U) -> f64 {
        self.approximator.evaluate(features).unwrap()
    }

    fn update_features(&mut self, features: &Features, _: &U, error: f64) {
        self.approximator.update(&mut self.optimiser, features, error).ok();
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Parameterised)]
pub struct StableCFA<P, B, O> {
    pub policy: P,
    pub basis: B,
    pub optimiser: O,
    #[weights] pub approximator: ScalarFunction,
}

impl<P: Parameterised, B: Projector, O: Optimiser> StableCFA<P, B, O> {
    pub fn new(policy: P, basis: B, optimiser: O) -> Self {
        let wd = policy.weights_dim();
        let bf = basis.n_features();
        let approximator = ScalarFunction::zeros(wd[0] * wd[1] + bf);

        StableCFA { policy, basis, optimiser, approximator }
    }

    pub fn to_lfa(&self) -> crate::fa::linear::LFA<B, O, ScalarFunction>
    where
        B: Clone,
        O: Clone,
    {
        let w = self.weights_view();
        let dim = self.weights_dim();
        let basis_rows = self.basis.n_features();

        let lfa_weights = w.slice(s![(dim[0] - basis_rows).., 0]);

        crate::fa::linear::LFA::new(
            self.basis.clone(),
            self.optimiser.clone(),
            ScalarFunction::new(lfa_weights.to_owned())
        )
    }
}

impl<P, B, O> StableCFA<P, B, O>
where
    P: Parameterised,
    B: Projector,
    O: Optimiser,
{
    pub fn evaluate_baseline<X: DerefSlice>(&self, state: &X) -> f64 where B: Clone {
        let wd = self.policy.weights_dim();

        let f_basis = self.basis.project(state.deref_slice()).unwrap();
        let f_policy = Features::from(vec![0.0; wd[0] * wd[1]]);

        let features = f_policy.stack(f_basis);

        self.approximator.evaluate(&features).unwrap()
    }
}

// Q(x, u):
impl<X, U, P, B, O> StateActionFunction<X, U> for StableCFA<P, B, O>
where
    X: DerefSlice,
    P: DifferentiablePolicy<X, Action = U>,
    B: Projector,
    O: Optimiser,
{
    type Output = f64;

    fn evaluate(&self, state: &X, action: &U) -> Self::Output {
        let features = self.features(state, action);

        self.approximator.evaluate(&features).unwrap()
    }

    fn update(&mut self, state: &X, action: &U, error: Self::Output) {
        let features = self.features(state, action);

        self.approximator.update(&mut self.optimiser, &features, error).ok();
    }
}

impl<X, U, B, P, O> DifferentiableStateActionFunction<X, U> for StableCFA<P, B, O>
where
    X: DerefSlice,
    P: DifferentiablePolicy<X, Action = U>,
    B: Projector,
    O: Optimiser,
{
    type Gradient = LFAGradient;

    fn grad(&self, state: &X, action: &U) -> Self::Gradient {
        LFAGradient::from_features(
            [self.n_features(), 1], 0,
            self.features(state, action)
        )
    }

    // fn zero_grad(&self) -> Self::Gradient {
        // LFAGradient::empty(self.weights_dim())
    // }
}

impl<X, U, B, P, O> LinearStateActionFunction<X, U> for StableCFA<P, B, O>
where
    X: DerefSlice,
    P: DifferentiablePolicy<X, Action = U>,
    B: Projector,
    O: Optimiser,
{
    fn n_features(&self) -> usize {
        let [r, c] = self.approximator.weights_dim();

        r * c
    }

    fn features(&self, state: &X, action: &U) -> Features {
        let gl_policy = self.policy.grad_log(state, action);
        let gl_policy = Features::Dense(gl_policy.index_axis_move(Axis(1), 0));

        gl_policy.stack(self.basis.project(state.deref_slice()).unwrap())
    }

    fn evaluate_features(&self, features: &Features, _: &U) -> f64 {
        self.approximator.evaluate(features).unwrap()
    }

    fn update_features(&mut self, features: &Features, _: &U, error: f64) {
        self.approximator.update(&mut self.optimiser, features, error).ok();
    }
}
