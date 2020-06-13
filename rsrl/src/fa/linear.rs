use crate::{
    fa::{GradientUpdate, ScaledGradientUpdate, StateActionUpdate, StateUpdate},
    params::*,
    Differentiable,
    Enumerable,
    Function,
    Handler,
};
use ndarray::{Array1, ArrayBase, Axis, DataMut, Dimension, Ix1, IntoDimension};

pub use lfa::*;

pub mod basis {
    pub use lfa::basis::*;

    use crate::params::Parameterised;
    use ndarray::Axis;

    #[derive(Clone, Debug)]
    #[cfg_attr(
        feature = "serde",
        derive(Serialize, Deserialize),
        serde(crate = "serde_crate")
    )]
    pub struct CompatibleBasis<P>(pub P);

    impl<P> spaces::Space for CompatibleBasis<P>
    where P: Parameterised
    {
        type Value = super::Features;

        fn dim(&self) -> spaces::Dim { spaces::Dim::Finite(self.0.n_weights()) }

        fn card(&self) -> spaces::Card { spaces::Card::Infinite }
    }

    impl<S, P> Basis<(S, P::Action)> for CompatibleBasis<P>
    where P: crate::policies::DifferentiablePolicy<S>
    {
        fn n_features(&self) -> usize { self.0.n_weights() }

        fn project(&self, args: (S, P::Action)) -> Result<super::Features, super::Error> {
            let gl = self.0.grad_log(args).into_shape(self.n_features()).unwrap();

            Ok(super::Features::Dense(gl))
        }
    }

    impl<P> Combinators for CompatibleBasis<P> {}

    /// Stable Compatible Basis
    #[derive(Clone, Debug)]
    #[cfg_attr(
        feature = "serde",
        derive(Serialize, Deserialize),
        serde(crate = "serde_crate")
    )]
    pub struct SCB<P, B> {
        pub policy: P,
        pub basis: B,
    }

    impl<P, B> spaces::Space for SCB<P, B>
    where
        P: Parameterised,
        B: spaces::Space,
    {
        type Value = super::Features;

        fn dim(&self) -> spaces::Dim {
            let bdim: usize = self.basis.dim().into();

            spaces::Dim::Finite(self.policy.n_weights() + bdim)
        }

        fn card(&self) -> spaces::Card { spaces::Card::Infinite }
    }

    impl<'s, S, A, P, B> Basis<(&'s S, A)> for SCB<P, B>
    where
        A: std::borrow::Borrow<P::Action>,
        P: crate::policies::DifferentiablePolicy<&'s S>,
        B: Basis<&'s S, Value = super::Features>,
    {
        fn n_features(&self) -> usize {
            let bdim: usize = self.basis.dim().into();

            self.policy.n_weights() + bdim
        }

        fn project(&self, (s, a): (&'s S, A)) -> Result<super::Features, super::Error> {
            let b = self.basis.project(s)?.into_dense().into_raw_vec();
            let gl = self
                .policy
                .grad_log((s, a.borrow()))
                .index_axis_move(Axis(1), 0)
                .into_raw_vec();

            Ok(super::Features::Dense(
                gl.into_iter().chain(b.into_iter()).collect(),
            ))
        }
    }

    impl<P, B> Combinators for SCB<P, B> {}
}

type Jacobian = Columnar<Features>;

impl Buffer for Features {
    type Dim = Ix1;

    fn dim(&self) -> usize { self.n_features() }

    fn n_dim(&self) -> usize { 1 }

    fn raw_dim(&self) -> Ix1 { ndarray::Ix1(self.n_features()) }

    fn addto<D: DataMut<Elem = f64>>(&self, arr: &mut ArrayBase<D, Ix1>) {
        Features::addto(self, arr)
    }

    fn scaled_addto<D: DataMut<Elem = f64>>(&self, alpha: f64, arr: &mut ArrayBase<D, Ix1>) {
        Features::scaled_addto(self, alpha, arr)
    }
}

impl BufferMut for Features {
    fn zeros<D: IntoDimension<Dim = Ix1>>(dim: D) -> Features {
        Features::Sparse(lfa::SparseActivations {
            dim: dim.into_dimension().into_pattern(),
            activations: ::std::collections::HashMap::new(),
        })
    }

    fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        match self {
            Features::Sparse(sa) => {
                if f(0.0).abs() > 1e-5 {
                    Features::Dense(Array1::from_shape_fn(sa.dim, |ref i| f(sa.activations[i])))
                } else {
                    Features::Sparse(SparseActivations {
                        dim: sa.dim,
                        activations: sa.iter().map(|(&k, &v)| (k, f(v))).collect(),
                    })
                }
            },
            Features::Dense(acts) => Features::Dense(acts.mapv(f)),
        }
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) { Features::map_inplace(self, f) }

    fn merge(&self, other: &Features, f: impl Fn(f64, f64) -> f64) -> Self {
        Features::merge(self, other, f)
    }

    fn merge_into(self, other: &Features, f: impl Fn(f64, f64) -> f64) -> Self {
        Features::merge_into(self, other, f)
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        Features::merge_inplace(self, other, f)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Common
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<I, J, B, D, O> Handler<GradientUpdate<J>> for LFA<B, ArrayBase<D, I>, O>
where
    I: Dimension,
    J: Buffer<Dim = I>,
    D: DataMut<Elem = f64>,
{
    type Response = ();
    type Error = Error;

    fn handle(&mut self, msg: GradientUpdate<J>) -> Result<()> {
        Ok(msg.0.addto(&mut self.weights))
    }
}

impl<I, J, B, D, O> Handler<ScaledGradientUpdate<J>> for LFA<B, ArrayBase<D, I>, O>
where
    I: Dimension,
    J: Buffer<Dim = I>,
    D: DataMut<Elem = f64>,
{
    type Response = ();
    type Error = Error;

    fn handle(&mut self, msg: ScaledGradientUpdate<J>) -> Result<()> {
        Ok(msg.jacobian.scaled_addto(msg.alpha, &mut self.weights))
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Implement V(s)
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<B, O> Parameterised for ScalarLFA<B, O> {
    fn weights_dim(&self) -> (usize, usize) { (self.weights.len(), 1) }

    fn weights(&self) -> Weights { self.weights.clone().insert_axis(Axis(1)) }

    fn weights_view(&self) -> WeightsView { self.weights.view().insert_axis(Axis(1)) }

    fn weights_view_mut(&mut self) -> WeightsViewMut {
        self.weights.view_mut().insert_axis(Axis(1))
    }
}

impl<S, B, O> Function<(S,)> for ScalarLFA<B, O>
where
    B: basis::Basis<S, Value = Features>,
    O: optim::Optimiser,
{
    type Output = f64;

    fn evaluate(&self, (s,): (S,)) -> f64 { self.evaluate(s).unwrap() }
}

impl<S, B, O> Differentiable<(S,)> for ScalarLFA<B, O>
where
    B: basis::Basis<S, Value = Features>,
    O: optim::Optimiser,
{
    type Jacobian = Features;

    fn grad(&self, (s,): (S,)) -> Features { self.basis.project(s).unwrap() }

    fn grad_log(&self, (s,): (S,)) -> Features {
        self.basis.project(s).unwrap().map_into(|x| 1.0 / x)
    }
}

impl<S, B, O> Handler<StateUpdate<S, f64>> for ScalarLFA<B, O>
where
    B: basis::Basis<S, Value = Features>,
    O: optim::Optimiser,
{
    type Response = ();
    type Error = Error;

    fn handle(&mut self, msg: StateUpdate<S, f64>) -> Result<()> {
        self.update(msg.state, msg.error)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Implement Q(s, a) with arbitrary a
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<S, A, B, O> Function<(S, A)> for ScalarLFA<B, O>
where
    B: basis::Basis<(S, A), Value = Features>,
    O: optim::Optimiser,
{
    type Output = f64;

    fn evaluate(&self, args: (S, A)) -> Self::Output { self.evaluate(args).unwrap() }
}

impl<S, A, B, O> Differentiable<(S, A)> for ScalarLFA<B, O>
where
    B: basis::Basis<(S, A), Value = Features>,
    O: optim::Optimiser,
{
    type Jacobian = Features;

    fn grad(&self, args: (S, A)) -> Features { self.basis.project(args).unwrap() }

    fn grad_log(&self, args: (S, A)) -> Features {
        self.basis.project(args).unwrap().map_into(|x| 1.0 / x)
    }
}

impl<S, A, B, O> Handler<StateActionUpdate<S, A, f64>> for ScalarLFA<B, O>
where
    B: basis::Basis<(S, A), Value = Features>,
    O: optim::Optimiser,
{
    type Response = ();
    type Error = Error;

    fn handle(&mut self, msg: StateActionUpdate<S, A, f64>) -> Result<()> {
        self.update((msg.state, msg.action), msg.error)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Implement Q(s, a) with discrete a
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<B, O> Parameterised for VectorLFA<B, O> {
    fn weights_dim(&self) -> (usize, usize) { self.weights.dim() }

    fn weights(&self) -> Weights { self.weights.clone() }

    fn weights_view(&self) -> WeightsView { self.weights.view() }

    fn weights_view_mut(&mut self) -> WeightsViewMut { self.weights.view_mut() }
}

impl<S, B, O> Function<(S,)> for VectorLFA<B, O>
where
    B: basis::Basis<S, Value = Features>,
    O: optim::Optimiser,
{
    type Output = Vec<f64>;

    fn evaluate(&self, (s,): (S,)) -> Self::Output { self.evaluate(s).unwrap().into_raw_vec() }
}

impl<S, T, B, O> Function<(S, T)> for VectorLFA<B, O>
where
    T: std::borrow::Borrow<usize>,
    B: basis::Basis<S, Value = Features>,
    O: optim::Optimiser,
{
    type Output = f64;

    fn evaluate(&self, (s, a): (S, T)) -> Self::Output {
        self.evaluate_index(s, *a.borrow()).unwrap()
    }
}

impl<S, T, B, O> Differentiable<(S, T)> for VectorLFA<B, O>
where
    T: std::borrow::Borrow<usize>,
    B: basis::Basis<S, Value = Features>,
    O: optim::Optimiser,
{
    type Jacobian = Jacobian;

    fn grad(&self, (s, a): (S, T)) -> Self::Jacobian {
        self.basis
            .project(s)
            .map(|b| Jacobian::from_column(self.weights.ncols(), *a.borrow(), b))
            .unwrap()
    }

    fn grad_log(&self, (s, a): (S, T)) -> Jacobian {
        self.basis
            .project(s)
            .map(|b| {
                let b = 1.0 / b.into_dense();

                Jacobian::from_column(self.weights.ncols(), *a.borrow(), Features::Dense(b))
            })
            .unwrap()
    }
}

impl<S, B, O> Enumerable<(S,)> for VectorLFA<B, O>
where
    B: basis::Basis<S, Value = Features>,
    O: optim::Optimiser,
{
    fn len(&self, _: (S,)) -> usize { self.weights.ncols() }

    fn evaluate_index(&self, (s,): (S,), index: usize) -> f64 {
        self.evaluate_index(s, index).unwrap()
    }
}

impl<S, B, O, E> Handler<StateUpdate<S, E>> for VectorLFA<B, O>
where
    B: basis::Basis<S, Value = Features>,
    O: optim::Optimiser,
    E: IntoIterator<Item = f64>,
{
    type Response = ();
    type Error = Error;

    fn handle(&mut self, msg: StateUpdate<S, E>) -> Result<()> {
        self.update(msg.state, msg.error)
    }
}

impl<S, A, B, O> Handler<StateActionUpdate<S, A, f64>> for VectorLFA<B, O>
where
    A: std::borrow::Borrow<usize>,
    B: basis::Basis<S, Value = Features>,
    O: optim::Optimiser,
{
    type Response = ();
    type Error = Error;

    fn handle(&mut self, msg: StateActionUpdate<S, A, f64>) -> Result<()> {
        self.update_index(msg.state, *msg.action.borrow(), msg.error)
    }
}
