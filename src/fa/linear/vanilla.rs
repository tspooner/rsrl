use crate::{
    core::DerefSlice,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction, DifferentiableStateFunction,
        StateActionFunction, DifferentiableStateActionFunction, EnumerableStateActionFunction,
        linear::{
            LFAGradient, Features,
            LinearStateFunction, LinearStateActionFunction,
            Approximator, ScalarApproximator, VectorApproximator,
            ScalarFunction, PairFunction, VectorFunction,
            basis::Projector,
            optim::Optimiser,
        }
    },
};

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Parameterised)]
pub struct LFA<B, O, A> {
    pub basis: B,
    pub optimiser: O,
    #[weights] pub approximator: A,
}

impl<B: Projector, O: Optimiser, A: Approximator> LFA<B, O, A> {
    pub fn new(basis: B, optimiser: O, approximator: A) -> Self {
        LFA {
            basis,
            optimiser,
            approximator,
        }
    }
}

impl<B: Projector, O: Optimiser> LFA<B, O, ScalarFunction> {
    pub fn scalar(basis: B, optimiser: O) -> Self {
        let approximator = ScalarFunction::zeros(basis.n_features());

        LFA::new(basis, optimiser, approximator)
    }
}

impl<B: Projector, O: Optimiser> LFA<B, O, PairFunction> {
    pub fn pair(basis: B, optimiser: O) -> Self {
        let approximator = PairFunction::zeros(basis.n_features());

        LFA::new(basis, optimiser, approximator)
    }
}

impl<B: Projector, O: Optimiser> LFA<B, O, VectorFunction> {
    pub fn vector(basis: B, optimiser: O, n_actions: usize) -> Self {
        let approximator = VectorFunction::zeros(basis.n_features(), n_actions);

        LFA::new(basis, optimiser, approximator)
    }
}

// V(s):
impl<X, B, O, A> StateFunction<X> for LFA<B, O, A>
where
    X: DerefSlice,
    B: Projector,
    O: Optimiser,
    A: Approximator,
{
    type Output = A::Output;

    fn evaluate(&self, state: &X) -> A::Output {
        let features = self.basis.project(state.deref_slice()).unwrap();

        self.approximator.evaluate(&features).unwrap()
    }

    fn update(&mut self, state: &X, error: A::Output) {
        self.approximator.update(
            &mut self.optimiser,
            &self.basis.project(state.deref_slice()).unwrap(),
            error
        ).ok();
    }
}

impl<X, B, O, A> DifferentiableStateFunction<X> for LFA<B, O, A>
where
    X: DerefSlice,
    B: Projector,
    O: Optimiser,
    A: ScalarApproximator,
{
    type Gradient = LFAGradient;

    fn grad(&self, state: &X) -> LFAGradient {
        LFAGradient::from_features(
            self.weights_dim(), 0,
            self.basis.project(state.deref_slice()).unwrap()
        )
    }
}

impl<X, B, O, A> LinearStateFunction<X> for LFA<B, O, A>
where
    X: DerefSlice,
    B: Projector,
    O: Optimiser,
    A: ScalarApproximator,
{
    fn n_features(&self) -> usize {
        self.basis.n_features()
    }

    fn features(&self, state: &X) -> Features {
        self.basis.project(state.deref_slice()).unwrap()
    }

    fn evaluate_features(&self, features: &Features) -> f64 {
        self.approximator.evaluate(features).unwrap()
    }

    fn update_features(&mut self, features: &Features, error: f64) {
        self.approximator.update(&mut self.optimiser, features, error).ok();
    }
}

// Q(x, u):
impl<X, B, O, A> StateActionFunction<X, usize> for LFA<B, O, A>
where
    X: DerefSlice,
    B: Projector,
    O: Optimiser,
    A: VectorApproximator,
{
    type Output = f64;

    fn evaluate(&self, state: &X, action: &usize) -> Self::Output {
        let features = self.basis.project(state.deref_slice()).unwrap();

        self.approximator.evaluate_index(&features, *action).unwrap()
    }

    fn update(&mut self, state: &X, action: &usize, error: Self::Output) {
        let features = self.basis.project(state.deref_slice()).unwrap();

        self.approximator.update_index(&mut self.optimiser, &features, *action, error).ok();
    }
}

impl<X, B, O, A> DifferentiableStateActionFunction<X, usize> for LFA<B, O, A>
where
    X: DerefSlice,
    B: Projector,
    O: Optimiser,
    A: VectorApproximator,
{
    type Gradient = LFAGradient;

    fn grad(&self, state: &X, action: &usize) -> Self::Gradient {
        LFAGradient::from_features(
            self.weights_dim(),
            *action,
            self.basis.project(state.deref_slice()).unwrap()
        )
    }
}

impl<X, B, O, A> EnumerableStateActionFunction<X> for LFA<B, O, A>
where
    X: DerefSlice,
    B: Projector,
    O: Optimiser,
    A: VectorApproximator,
{
    fn n_actions(&self) -> usize {
        self.approximator.n_outputs()
    }

    fn evaluate_all(&self, state: &X) -> Vec<f64> {
        self.approximator.evaluate(&self.basis.project(state.deref_slice()).unwrap()).unwrap()
    }

    fn update_all(&mut self, state: &X, errors: Vec<f64>) {
        self.approximator.update(
            &mut self.optimiser,
            &self.basis.project(state.deref_slice()).unwrap(),
            errors
        ).ok();
    }
}

impl<X, B, O, A> LinearStateActionFunction<X, usize> for LFA<B, O, A>
where
    X: DerefSlice,
    B: Projector,
    O: Optimiser,
    A: VectorApproximator,
{
    fn n_features(&self) -> usize {
        self.basis.n_features()
    }

    fn features(&self, state: &X, _: &usize) -> Features {
        self.basis.project(state.deref_slice()).unwrap()
    }

    fn evaluate_features(&self, features: &Features, action: &usize) -> f64 {
        self.approximator.evaluate_index(features, *action).unwrap()
    }

    fn update_features(&mut self, features: &Features, action: &usize, error: f64) {
        self.approximator.update_index(&mut self.optimiser, features, *action, error).ok();
    }
}
