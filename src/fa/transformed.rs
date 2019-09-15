use crate::{
    core::DerefSlice,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction, DifferentiableStateFunction,
        StateActionFunction, DifferentiableStateActionFunction, EnumerableStateActionFunction,
        linear::{
            Approximator, ScalarApproximator, VectorApproximator,
            ScalarFunction, PairFunction, VectorFunction,
            Features,
            basis::Projector,
            optim::SGD,
        },
        transforms,
    },
    linalg::Columnar,
};
use std::ops::{Deref, AddAssign};

/// Transformed linear function approximator.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Parameterised)]
pub struct TransformedLFA<B, A, T = transforms::Identity> {
    pub basis: B,
    #[weights] pub approximator: A,
    pub transform: T,
}

impl<B, A, T> TransformedLFA<B, A, T>
where
    B: Projector,
    A: Approximator,
    T: transforms::Transform<A::Output>,
{
    pub fn new(basis: B, approximator: A, transform: T) -> Self {
        TransformedLFA {
            basis,
            approximator,
            transform,
        }
    }
}

impl<B, T> TransformedLFA<B, ScalarFunction, T>
where
    B: Projector,
    T: transforms::Transform<<ScalarFunction as Approximator>::Output>,
{
    pub fn scalar(basis: B, transform: T) -> Self {
        let approximator = ScalarFunction::zeros(basis.n_features());

        TransformedLFA::new(basis, approximator, transform)
    }
}

impl<B, T> TransformedLFA<B, PairFunction, T>
where
    B: Projector,
    T: transforms::Transform<<PairFunction as Approximator>::Output>,
{
    pub fn pair(basis: B, transform: T) -> Self {
        let approximator = PairFunction::zeros(basis.n_features());

        TransformedLFA::new(basis, approximator, transform)
    }
}

impl<B, T> TransformedLFA<B, VectorFunction, T>
where
    B: Projector,
    T: transforms::Transform<<VectorFunction as Approximator>::Output>,
{
    pub fn vector(basis: B, n_actions: usize, transform: T) -> Self {
        let approximator = VectorFunction::zeros(basis.n_features(), n_actions);

        TransformedLFA::new(basis, approximator, transform)
    }
}

// V(s):
impl<X, B, A, T> StateFunction<X> for TransformedLFA<B, A, T>
where
    X: DerefSlice,
    B: Projector,
    A: ScalarApproximator,
    T: transforms::Transform<A::Output>,
{
    type Output = T::Output;

    fn evaluate(&self, state: &X) -> T::Output {
        self.basis.project(state.deref_slice())
            .and_then(|f| self.approximator.evaluate(&f))
            .map(|v| self.transform.transform(v))
            .unwrap()
    }

    fn update(&mut self, state: &X, error: T::Output) {
        let (f, df) = self.basis.project(state.deref_slice())
            .and_then(|f| self.approximator.evaluate(&f).map(|v| (f, v)))
            .map(|(f, v)| (f, self.transform.grad_scaled(v, error)))
            .unwrap();

        self.approximator.update(&mut SGD(1.0), &f, df).ok();
    }
}

impl<X, B, A, T> DifferentiableStateFunction<X> for TransformedLFA<B, A, T>
where
    X: DerefSlice,
    B: Projector,
    A: ScalarApproximator,
    T: transforms::Transform<A::Output, Output = f64>,
{
    type Gradient = Columnar;

    fn grad(&self, state: &X) -> Self::Gradient {
        self.basis.project(state.deref_slice())
            .and_then(|f| self.approximator.evaluate(&f).map(|v| (f, v)))
            .map(|(f, v)| Columnar::from_column(
                1, 0, f.expanded().mapv(|x| self.transform.grad_scaled(v, x))
            ))
            .unwrap()
    }
}

// // Q(x, u):
// impl<X, B, A> StateActionFunction<X, usize> for TransformedLFA<B, A>
// where
    // X: DerefSlice,
    // B: Projector,
    // A: VectorApproximator,
// {
    // type Output = f64;

    // fn evaluate(&self, state: &X, action: &usize) -> Self::Output {
        // let features = self.basis.project(state.deref_slice());

        // self.approximator.evaluate_index(&features, *action).unwrap()
    // }

    // fn update(&mut self, state: &X, action: &usize, error: Self::Output) {
        // let features = self.basis.project(state.deref_slice());

        // self.approximator.update_index(&features, *action, error).ok();
    // }
// }

// impl<X, B, A> DifferentiableStateActionFunction<X, usize> for TransformedLFA<B, A>
// where
    // X: DerefSlice,
    // B: Projector,
    // A: VectorApproximator,
// {
    // fn grad(&self, state: &X, _: &usize) -> Matrix<f64> {
        // // TODO|XXX: This should be replaced with `Gradient` types because the gradient here is
        // // actually a matrix with `n_actions` number of columns. The problem is that they're
        // // typically all empty which we don't wan't to populate.
        // self.basis.project(state.deref_slice()).into_vector().insert_axis(Axis(1))
    // }

    // fn update_grad(&mut self, grad: &Matrix<f64>) {
        // self.approximator.weights_view_mut().add_assign(grad);
    // }

    // fn update_grad_scaled(&mut self, grad: &Matrix<f64>, error: f64) {
        // self.approximator.weights_view_mut().scaled_add(error, grad);
    // }
// }

// impl<X, B, A> EnumerableStateActionFunction<X> for TransformedLFA<B, A>
// where
    // X: DerefSlice,
    // B: Projector,
    // A: VectorApproximator,
// {
    // fn n_actions(&self) -> usize {
        // self.approximator.n_outputs()
    // }

    // fn evaluate_all(&self, state: &X) -> Vector<f64> {
        // self.approximator.evaluate(&self.basis.project(state.deref_slice())).unwrap()
    // }

    // fn update_all(&mut self, state: &X, errors: Vector<f64>) {
        // self.approximator.update(&self.basis.project(state.deref_slice()), errors).ok();
    // }
// }
