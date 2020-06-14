use super::{BuilderSupport, Gaussian};
use crate::{
    fa::{GradientUpdate, ScaledGradientUpdate, StateActionUpdate},
    params::*,
    policies::Policy,
    spaces::Space,
    Differentiable,
    Function,
    Handler,
};
use ndarray::{Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use rand::Rng;
use rstat::{
    builder::{BuildNormal, Builder},
    fitting::Score,
    normal,
    ContinuousDistribution,
    Distribution,
};

impl<M> Parameterised for Gaussian<M, f64>
where M: Parameterised
{
    fn weights(&self) -> Weights { self.mean.weights() }

    fn weights_view(&self) -> WeightsView { self.mean.weights_view() }

    fn weights_view_mut(&mut self) -> WeightsViewMut { self.mean.weights_view_mut() }

    fn weights_dim(&self) -> (usize, usize) { self.mean.weights_dim() }
}

impl<'x, X, A, M> Function<(&'x X, A)> for Gaussian<M, f64>
where
    A: std::borrow::Borrow<M::Output>,
    M: Function<(&'x X,)>,

    M::Output: Clone,

    Builder: BuildNormal<M::Output, f64>,
    BuilderSupport<M::Output, f64>: Space<Value = M::Output>,
{
    type Output = f64;

    fn evaluate(&self, (x, a): (&'x X, A)) -> f64 {
        Builder::build_unchecked(self.compute_mean(x), self.stddev).pdf(a.borrow())
    }
}

impl<'x, X, A, M> Differentiable<(&'x X, A)> for Gaussian<M, f64>
where
    A: std::borrow::Borrow<f64>,

    M: Parameterised + Differentiable<(&'x X,), Output = f64>,

    M::Jacobian: Buffer<Dim = Ix1>,

    Builder: BuildNormal<M::Output, f64>,
    BuilderSupport<M::Output, f64>: Space<Value = M::Output>,
{
    type Jacobian = Array2<f64>;

    fn grad(&self, _: (&'x X, A)) -> Array2<f64> { todo!() }

    fn grad_log(&self, (x, a): (&'x X, A)) -> Array2<f64> {
        let dist = Builder::build_unchecked(self.compute_mean(x), self.stddev);
        let grad_mean = self.mean.grad((x,)).into_dense().insert_axis(Axis(1));

        let normal::Grad { mu: gl_mean, .. } = dist.score(&[*a.borrow()]);

        grad_mean * gl_mean
    }
}

impl<'x, X, M> Policy<&'x X> for Gaussian<M, f64>
where
    M: Function<(&'x X,)>,

    M::Output: Clone,

    Builder: BuildNormal<M::Output, f64>,
    BuilderSupport<M::Output, f64>: Space<Value = M::Output>,
{
    type Action = M::Output;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, x: &'x X) -> Self::Action {
        Builder::build_unchecked(self.compute_mean(x), self.stddev).sample(rng)
    }

    fn mode(&self, x: &'x X) -> Self::Action { self.compute_mean(x) }
}

impl<'x, X, A, M> Handler<StateActionUpdate<&'x X, A>> for Gaussian<M, f64>
where
    A: std::borrow::Borrow<f64>,

    M: Parameterised + Differentiable<(&'x X,), Output = f64>,

    M::Jacobian: Buffer<Dim = Ix1>,

    Builder: BuildNormal<M::Output, f64>,
    BuilderSupport<M::Output, f64>: Space<Value = M::Output>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: StateActionUpdate<&'x X, A>) -> Result<Self::Response, Self::Error> {
        let dist = Builder::build_unchecked(self.compute_mean(msg.state), self.stddev);
        let normal::Grad { mu: gl_mean, .. } = dist.score(&[*msg.action.borrow()]);

        self.mean.grad((msg.state,)).scaled_addto(
            msg.error * gl_mean,
            &mut self.mean.weights_view_mut().column_mut(0),
        );

        Ok(())
    }
}

impl<'m, M, D> Handler<GradientUpdate<ArrayBase<D, Ix2>>> for Gaussian<M, f64>
where
    M: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: GradientUpdate<ArrayBase<D, Ix2>>) -> Result<Self::Response, Self::Error> {
        self.handle(GradientUpdate(&msg.0))
    }
}

impl<'m, M, D> Handler<GradientUpdate<&'m ArrayBase<D, Ix2>>> for Gaussian<M, f64>
where
    M: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: GradientUpdate<&'m ArrayBase<D, Ix2>>) -> Result<Self::Response, Self::Error> {
        msg.0.addto(&mut self.mean.weights_view_mut());

        Ok(())
    }
}

impl<'m, M, D> Handler<ScaledGradientUpdate<ArrayBase<D, Ix2>>> for Gaussian<M, f64>
where
    M: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: ScaledGradientUpdate<ArrayBase<D, Ix2>>) -> Result<Self::Response, Self::Error> {
        self.handle(ScaledGradientUpdate {
            alpha: msg.alpha,
            jacobian: &msg.jacobian,
        })
    }
}

impl<'m, M, D> Handler<ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>> for Gaussian<M, f64>
where
    M: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>) -> Result<Self::Response, Self::Error> {
        msg.jacobian
            .scaled_addto(msg.alpha, &mut self.mean.weights_view_mut());

        Ok(())
    }
}
