use crate::{
    fa::{GradientUpdate, ScaledGradientUpdate, StateActionUpdate},
    params::{Parameterised, Weights, WeightsView, WeightsViewMut},
    policies::{DifferentiablePolicy, Policy},
    Differentiable,
    Function,
    Handler,
};
use ndarray::{Array2, ArrayBase, ArrayView2, Axis, Data, Ix2};
use rand::Rng;

/// Independent Policy Pair (IPP).
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct IPP<P1, P2>(pub P1, pub P2);

impl<P1: Parameterised, P2: Parameterised> Parameterised for IPP<P1, P2> {
    fn weights(&self) -> Weights { stack![Axis(1), self.0.weights(), self.1.weights()] }

    fn weights_view(&self) -> WeightsView { unimplemented!() }

    fn weights_view_mut(&mut self) -> WeightsViewMut { unimplemented!() }

    fn weights_dim(&self) -> (usize, usize) {
        let d0 = self.0.weights_dim();
        let d1 = self.1.weights_dim();

        (d0.0.max(d1.0), d0.1 + d1.1)
    }
}

impl<'s, S, A, P1, P2> Function<(&'s S, A)> for IPP<P1, P2>
where
    A: std::borrow::Borrow<(P1::Action, P2::Action)>,
    P1: Policy<&'s S>,
    P2: Policy<&'s S>,
{
    type Output = f64;

    fn evaluate(&self, (s, a): (&'s S, A)) -> f64 {
        let a = a.borrow();

        self.0.evaluate((s, &a.0)) * self.1.evaluate((s, &a.1))
    }
}

fn combine(mut gl_0: Array2<f64>, mut gl_1: Array2<f64>) -> Array2<f64> {
    let nr_0 = gl_0.nrows();
    let nr_1 = gl_1.nrows();

    fn resize(gl: Array2<f64>, n_rows: usize) -> Array2<f64> {
        let gl_rows = gl.nrows();

        let mut new_gl = unsafe { Array2::uninitialized((n_rows, gl.ncols())) };

        new_gl.slice_mut(s![0..gl_rows, ..]).assign(&gl);
        new_gl.slice_mut(s![gl_rows.., ..]).fill(0.0);

        new_gl
    }

    if nr_0 > nr_1 {
        gl_1 = resize(gl_1, nr_0);
    } else if nr_0 < nr_1 {
        gl_0 = resize(gl_0, nr_1);
    }

    stack![Axis(1), gl_0, gl_1]
}

impl<'s, S, A, P1, P2> Differentiable<(&'s S, A)> for IPP<P1, P2>
where
    A: std::borrow::Borrow<(P1::Action, P2::Action)>,
    P1: DifferentiablePolicy<&'s S>,
    P2: DifferentiablePolicy<&'s S>,
{
    type Jacobian = Array2<f64>;

    fn grad(&self, (s, a): (&'s S, A)) -> Array2<f64> {
        let a = a.borrow();

        let g_0 = self.0.grad((s, &a.0));
        let g_1 = self.1.grad((s, &a.1));

        combine(g_0, g_1)
    }

    fn grad_log(&self, (s, a): (&'s S, A)) -> Array2<f64> {
        let a = a.borrow();

        let gl_0 = self.0.grad_log((s, &a.0));
        let gl_1 = self.1.grad_log((s, &a.1));

        combine(gl_0, gl_1)
    }
}

impl<'s, S, P1, P2> Policy<&'s S> for IPP<P1, P2>
where
    P1: Policy<&'s S>,
    P2: Policy<&'s S>,
{
    type Action = (P1::Action, P2::Action);

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, s: &'s S) -> Self::Action {
        (self.0.sample(rng, s), self.1.sample(rng, s))
    }

    fn mode(&self, s: &'s S) -> Self::Action { (self.0.mode(s), self.1.mode(s)) }
}

impl<'s, S, A, P1, P2> Handler<StateActionUpdate<&'s S, A>> for IPP<P1, P2>
where
    A: std::borrow::Borrow<(P1::Action, P2::Action)>,
    P1: Policy<(&'s S,)>
        + for<'a> Handler<StateActionUpdate<&'s S, &'a <P1 as Policy<(&'s S,)>>::Action>>,
    P2: Policy<(&'s S,)>
        + for<'a> Handler<StateActionUpdate<&'s S, &'a <P2 as Policy<(&'s S,)>>::Action>>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: StateActionUpdate<&'s S, A>) -> Result<(), ()> {
        let a = msg.action.borrow();

        self.0
            .handle(StateActionUpdate {
                state: msg.state,
                action: &a.0,
                error: msg.error,
            })
            .ok();
        self.1
            .handle(StateActionUpdate {
                state: msg.state,
                action: &a.1,
                error: msg.error,
            })
            .ok();

        Ok(())
    }
}

impl<'m, D, P1, P2> Handler<GradientUpdate<&'m ArrayBase<D, Ix2>>> for IPP<P1, P2>
where
    D: Data<Elem = f64>,
    P1: Parameterised + Handler<GradientUpdate<ArrayView2<'m, f64>>>,
    P2: Parameterised + Handler<GradientUpdate<ArrayView2<'m, f64>>>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: GradientUpdate<&'m ArrayBase<D, Ix2>>) -> Result<(), ()> {
        let d0 = self.0.weights_dim();
        let d1 = self.1.weights_dim();

        self.0
            .handle(GradientUpdate(msg.0.slice(s![0..d0.0, 0..d0.1])))
            .ok();
        self.1
            .handle(GradientUpdate(msg.0.slice(s![0..d1.0, d0.1..])))
            .ok();

        Ok(())
    }
}

impl<'m, D, P1, P2> Handler<ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>> for IPP<P1, P2>
where
    D: Data<Elem = f64>,
    P1: Parameterised + Handler<ScaledGradientUpdate<ArrayView2<'m, f64>>>,
    P2: Parameterised + Handler<ScaledGradientUpdate<ArrayView2<'m, f64>>>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>) -> Result<(), ()> {
        let d0 = self.0.weights_dim();
        let d1 = self.1.weights_dim();

        self.0
            .handle(ScaledGradientUpdate {
                alpha: msg.alpha,
                jacobian: msg.jacobian.slice(s![0..d0.0, 0..d0.1]),
            })
            .ok();
        self.1
            .handle(ScaledGradientUpdate {
                alpha: msg.alpha,
                jacobian: msg.jacobian.slice(s![0..d1.0, d0.1..]),
            })
            .ok();

        Ok(())
    }
}
