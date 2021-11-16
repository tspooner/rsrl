use super::{BuilderDist, BuilderSupport, Gaussian};
use crate::{
    fa::{GradientUpdate, ScaledGradientUpdate, StateActionUpdate},
    params::*,
    policies::Policy,
    spaces::Space,
    Differentiable, Function, Handler,
};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use rand::Rng;
use rstat::{
    builder::{BuildNormal, Builder},
    fitting::Score,
    normal, ContinuousDistribution, Distribution,
};

impl<M, S> Parameterised for Gaussian<M, S>
where
    M: Parameterised,
    S: Parameterised,
{
    fn weights(&self) -> Weights {
        let w_mean = self.mean.weights();
        let w_stddev = self.stddev.weights();

        if w_stddev.is_empty() {
            w_mean
        } else {
            stack![Axis(0), w_mean, w_stddev]
        }
    }

    fn weights_view(&self) -> WeightsView {
        unimplemented!()
    }

    fn weights_view_mut(&mut self) -> WeightsViewMut {
        unimplemented!()
    }

    fn weights_dim(&self) -> (usize, usize) {
        let (rm, cm) = self.mean.weights_dim();
        let (rs, cs) = self.stddev.weights_dim();

        (rm + rs, cm.max(cs))
    }
}

impl<'x, X, A, M, S> Function<(&'x X, A)> for Gaussian<M, S>
where
    A: std::borrow::Borrow<M::Output>,

    M: Function<(&'x X,)>,
    S: Function<(&'x X,)>,

    M::Output: Clone,
    S::Output: std::ops::Add<f64, Output = S::Output> + super::IntoCov,

    Builder: BuildNormal<M::Output, S::Output>,
    BuilderSupport<M::Output, S::Output>: Space<Value = M::Output>,
{
    type Output = f64;

    fn evaluate(&self, (x, a): (&'x X, A)) -> f64 {
        self.dist(x).pdf(a.borrow())
    }
}

// impl<'x, X, A, M, S> Differentiable<(&'x X, A)> for Gaussian<M, S>
// where
// A: std::borrow::Borrow<M::Output>,

// M: Parameterised + Differentiable<(&'x X,)>,
// S: Parameterised + Differentiable<(&'x X,)>,

// M::Output: Clone,
// S::Output: std::ops::Add<f64, Output = S::Output>,

// Builder: BuildNormal<M::Output, S::Output>,
// BuilderDist<M::Output, S::Output>: Score<Grad = normal::Grad<M::Output, S::Output>>,
// BuilderSupport<M::Output, S::Output>: Space<Value = M::Output>,
// {
// type Jacobian = Array2<f64>;

// fn grad(&self, _: (&'x X, A)) -> Array2<f64> { todo!() }

// fn grad_log(&self, (x, a): (&'x X, A)) -> Array2<f64> {
// let grad_mean = self.mean.grad((x,)).into_dense();
// let ndim_mean = grad_mean.ndim();

// let grad_stddev = self.stddev.grad((x,)).into_dense();
// let ndim_stddev = grad_stddev.ndim();

// let normal::Grad {
// mu: gl_mean,
// Sigma: gl_stddev,
// } = self.dist(x).score(std::slice::from_ref(a.borrow()));

// match (ndim_mean, ndim_stddev) {
// (2, 2) => {
// let grad_mean: Array2<f64> = grad_mean.into_dimensionality().unwrap();
// let grad_stddev: Array2<f64> = grad_stddev.into_dimensionality().unwrap();

// stack![Axis(0), grad_mean * gl_mean, grad_stddev * gl_stddev]
// },
// (2, 1) => {
// let dim = (grad_stddev.shape()[0], grad_mean.shape()[1]);
// let grad_mean: Array2<f64> = grad_mean.into_dimensionality().unwrap();

// stack![
// Axis(0),
// grad_mean,
// grad_stddev.insert_axis(Axis(1)).broadcast(dim).unwrap()
// ]
// },
// (1, 1) => {
// let mut jac = grad_mean.into_raw_vec();
// let mut grad_stddev = grad_stddev.into_raw_vec();

// jac.append(&mut grad_stddev);

// Array1::from(jac).insert_axis(Axis(1))
// },
// _ => unimplemented!(),
// }
// }
// }

impl<'x, X, A, M, S> Differentiable<(&'x X, A)> for Gaussian<M, S>
where
    A: std::borrow::Borrow<M::Output>,

    M: Parameterised + Differentiable<(&'x X,), Output = f64>,
    S: Parameterised + Differentiable<(&'x X,), Output = f64>,

    Builder: BuildNormal<M::Output, S::Output>,
    BuilderDist<M::Output, S::Output>: Score<Grad = normal::Grad<M::Output, S::Output>>,
    BuilderSupport<M::Output, S::Output>: Space<Value = M::Output>,
{
    type Jacobian = Array2<f64>;

    fn grad(&self, _: (&'x X, A)) -> Array2<f64> {
        todo!()
    }

    fn grad_log(&self, (x, a): (&'x X, A)) -> Array2<f64> {
        let grad_mean = self.mean.grad((x,)).into_dense();
        let grad_stddev = self.stddev.grad((x,)).into_dense();

        let normal::Grad {
            mu: gl_mean,
            Sigma: gl_stddev,
        } = self.dist(x).score(std::slice::from_ref(a.borrow()));

        grad_mean
            .iter()
            .map(|x| x * gl_mean)
            .chain(grad_stddev.iter().map(|x| x * gl_stddev))
            .collect::<Array1<f64>>()
            .insert_axis(Axis(1))
    }
}

impl<'x, X, M, S> Policy<&'x X> for Gaussian<M, S>
where
    M: Function<(&'x X,)>,
    S: Function<(&'x X,)>,

    M::Output: Clone,
    S::Output: std::ops::Add<f64, Output = S::Output> + super::IntoCov,

    Builder: BuildNormal<M::Output, S::Output>,
    BuilderSupport<M::Output, S::Output>: Space<Value = M::Output>,
{
    type Action = M::Output;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, x: &'x X) -> Self::Action {
        self.dist(x).sample(rng)
    }

    fn mode(&self, x: &'x X) -> Self::Action {
        self.compute_mean(x)
    }
}

impl<'x, X, A, M, S> Handler<StateActionUpdate<&'x X, A>> for Gaussian<M, S>
where
    A: std::borrow::Borrow<f64>,

    M: Parameterised + Differentiable<(&'x X,), Output = f64>,
    S: Parameterised + Differentiable<(&'x X,), Output = f64>,

    M::Jacobian: Buffer<Dim = Ix1>,
    S::Jacobian: Buffer<Dim = Ix1>,

    Builder: BuildNormal<M::Output, S::Output>,
    BuilderSupport<M::Output, S::Output>: Space<Value = M::Output>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: StateActionUpdate<&'x X, A>) -> Result<Self::Response, Self::Error> {
        let normal::Grad {
            mu: gl_mean,
            Sigma: gl_stddev,
        } = self.dist(msg.state).score(&[*msg.action.borrow()]);

        self.mean.grad((msg.state,)).scaled_addto(
            msg.error * gl_mean,
            &mut self.mean.weights_view_mut().column_mut(0),
        );
        self.stddev.grad((msg.state,)).scaled_addto(
            msg.error * gl_stddev,
            &mut self.stddev.weights_view_mut().column_mut(0),
        );

        Ok(())
    }
}

impl<'m, M, S, D> Handler<GradientUpdate<ArrayBase<D, Ix2>>> for Gaussian<M, S>
where
    M: Parameterised,
    S: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(
        &mut self,
        msg: GradientUpdate<ArrayBase<D, Ix2>>,
    ) -> Result<Self::Response, Self::Error> {
        self.handle(GradientUpdate(&msg.0))
    }
}

impl<'m, M, S, D> Handler<GradientUpdate<&'m ArrayBase<D, Ix2>>> for Gaussian<M, S>
where
    M: Parameterised,
    S: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(
        &mut self,
        msg: GradientUpdate<&'m ArrayBase<D, Ix2>>,
    ) -> Result<Self::Response, Self::Error> {
        let n_mean = self.mean.n_weights();
        let n_stddev = self.stddev.n_weights();

        msg.0
            .slice(s![0..n_mean, ..])
            .addto(&mut self.mean.weights_view_mut());
        msg.0
            .slice(s![n_mean..(n_stddev + n_mean), ..])
            .addto(&mut self.stddev.weights_view_mut());

        Ok(())
    }
}

impl<'m, M, S, D> Handler<ScaledGradientUpdate<ArrayBase<D, Ix2>>> for Gaussian<M, S>
where
    M: Parameterised,
    S: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(
        &mut self,
        msg: ScaledGradientUpdate<ArrayBase<D, Ix2>>,
    ) -> Result<Self::Response, Self::Error> {
        self.handle(ScaledGradientUpdate {
            alpha: msg.alpha,
            jacobian: &msg.jacobian,
        })
    }
}

impl<'m, M, S, D> Handler<ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>> for Gaussian<M, S>
where
    M: Parameterised,
    S: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(
        &mut self,
        msg: ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>,
    ) -> Result<Self::Response, Self::Error> {
        let c = msg.jacobian.column(0);
        let n_mean = self.mean.n_weights();
        let n_stddev = self.stddev.n_weights();

        c.slice(s![0..n_mean])
            .insert_axis(Axis(1))
            .scaled_addto(msg.alpha, &mut self.mean.weights_view_mut());
        c.slice(s![n_mean..(n_stddev + n_mean)])
            .insert_axis(Axis(1))
            .scaled_addto(msg.alpha, &mut self.stddev.weights_view_mut());

        Ok(())
    }
}
