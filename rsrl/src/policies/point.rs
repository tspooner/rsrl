use crate::{
    fa::{GradientUpdate, ScaledGradientUpdate, StateActionUpdate},
    params::*,
    policies::Policy,
    Differentiable,
    Function,
    Handler,
};
use ndarray::{ArrayBase, ArrayView2, Data, Ix2};
use rand::Rng;

#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Point<F>(pub F);

impl<F> Point<F> {
    pub fn new(fa: F) -> Self { Point(fa) }
}

impl<'s, S, A, F> Function<(S, A)> for Point<F>
where
    A: std::borrow::Borrow<F::Output>,
    F: Function<(S,)>,
    F::Output: PartialEq,
{
    type Output = f64;

    fn evaluate(&self, (s, a): (S, A)) -> f64 {
        let v = self.0.evaluate((s,));

        if v.eq(a.borrow()) {
            1.0
        } else {
            0.0
        }
    }
}

impl<'s, S, A, F> Differentiable<(S, A)> for Point<F>
where
    A: std::borrow::Borrow<<F as Function<(S,)>>::Output>,
    F: Function<(S,)> + Differentiable<(S, A)>,

    <F as Function<(S,)>>::Output: PartialEq,
{
    type Jacobian = F::Jacobian;

    fn grad(&self, (s, a): (S, A)) -> F::Jacobian { self.0.grad((s, a)) }

    fn grad_log(&self, (s, a): (S, A)) -> F::Jacobian { self.0.grad_log((s, a)) }
}

impl<S, F> Policy<S> for Point<F>
where
    F: Function<(S,)>,
    F::Output: PartialEq,
{
    type Action = F::Output;

    fn sample<R: Rng + ?Sized>(&self, _: &mut R, s: S) -> Self::Action { self.0.evaluate((s,)) }

    fn mode(&self, s: S) -> Self::Action { self.0.evaluate((s,)) }
}

impl<S, A, F> Handler<StateActionUpdate<S, A>> for Point<F>
where
    A: std::borrow::Borrow<f64>,
    F: for<'s> Function<(&'s S,), Output = f64> + Handler<StateActionUpdate<S, A>>,
{
    type Response = F::Response;
    type Error = F::Error;

    fn handle(&mut self, msg: StateActionUpdate<S, A>) -> Result<Self::Response, Self::Error> {
        let mode = self.0.evaluate((&msg.state,));
        let error = (msg.action.borrow() - mode) * msg.error;

        self.0.handle(StateActionUpdate {
            state: msg.state,
            action: msg.action,
            error,
        })
    }
}

impl<'m, D, F> Handler<GradientUpdate<&'m ArrayBase<D, Ix2>>> for Point<F>
where
    D: Data<Elem = f64>,
    F: Parameterised + Handler<GradientUpdate<ArrayView2<'m, f64>>>,
{
    type Response = F::Response;
    type Error = F::Error;

    fn handle(
        &mut self,
        msg: GradientUpdate<&'m ArrayBase<D, Ix2>>,
    ) -> Result<F::Response, F::Error>
    {
        let dim = self.0.weights_dim();

        self.0.handle(GradientUpdate(msg.0.slice(s![0..dim.0, 0..dim.1])))
    }
}

impl<'m, D, F> Handler<ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>> for Point<F>
where
    D: Data<Elem = f64>,
    F: Parameterised + Handler<ScaledGradientUpdate<ArrayView2<'m, f64>>>,
{
    type Response = F::Response;
    type Error = F::Error;

    fn handle(
        &mut self,
        msg: ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>,
    ) -> Result<F::Response, F::Error>
    {
        let dim = self.0.weights_dim();

        self.0.handle(ScaledGradientUpdate {
            alpha: msg.alpha,
            jacobian: msg.jacobian.slice(s![0..dim.0, 0..dim.1]),
        })
    }
}
