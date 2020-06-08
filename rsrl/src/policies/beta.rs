use crate::{
    fa::{GradientUpdate, ScaledGradientUpdate, StateUpdate, StateActionUpdate},
    params::*,
    policies::Policy,
    Differentiable,
    Function,
    Handler,
};
use ndarray::{Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2};
use rand::Rng;
use rstat::{
    fitting::Score,
    statistics::{Modes, UnivariateMoments},
    univariate::beta,
    ContinuousDistribution,
    Distribution,
};

const MIN_TOL: f64 = 1.0;

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Beta<A, B = A> {
    pub alpha: A,
    pub beta: B,
}

impl<A, B> Beta<A, B> {
    pub fn new(alpha: A, beta: B) -> Self { Beta { alpha, beta } }

    #[inline]
    pub fn compute_alpha<S>(&self, s: S) -> f64
    where A: Function<(S,), Output = f64> {
        self.alpha.evaluate((s,)) + MIN_TOL
    }

    #[inline]
    pub fn compute_beta<S>(&self, s: S) -> f64
    where B: Function<(S,), Output = f64> {
        self.beta.evaluate((s,)) + MIN_TOL
    }

    #[inline]
    fn dist<'s, S>(&self, s: &'s S) -> beta::Beta
    where
        A: Function<(&'s S,), Output = f64>,
        B: Function<(&'s S,), Output = f64>,
    {
        beta::Beta::new_unchecked(self.compute_alpha(s), self.compute_beta(s))
    }
}

impl<A: Parameterised, B: Parameterised> Parameterised for Beta<A, B> {
    fn weights(&self) -> Weights { stack![Axis(0), self.alpha.weights(), self.beta.weights()] }

    fn weights_view(&self) -> WeightsView { unimplemented!() }

    fn weights_view_mut(&mut self) -> WeightsViewMut { unimplemented!() }

    fn weights_dim(&self) -> (usize, usize) {
        let (ra, _) = self.alpha.weights_dim();
        let (rb, _) = self.beta.weights_dim();

        (ra + rb, 1)
    }
}

impl<'s, S, U, A, B> Function<(&'s S, U)> for Beta<A, B>
where
    U: std::borrow::Borrow<f64>,
    A: Function<(&'s S,), Output = f64>,
    B: Function<(&'s S,), Output = f64>,
{
    type Output = f64;

    fn evaluate(&self, (s, a): (&'s S, U)) -> f64 { self.dist(s).pdf(a.borrow()) }
}

impl<'s, S, U, A, B> Differentiable<(&'s S, U)> for Beta<A, B>
where
    U: std::borrow::Borrow<f64>,

    A: Differentiable<(&'s S,), Output = f64>,
    B: Differentiable<(&'s S,), Output = f64>,

    A::Jacobian: Buffer<Dim = Ix1>,
    B::Jacobian: Buffer<Dim = Ix1>,
{
    type Jacobian = Array2<f64>;

    fn grad(&self, _: (&'s S, U)) -> Array2<f64> { todo!() }

    fn grad_log(&self, (s, a): (&'s S, U)) -> Array2<f64> {
        let grad_alpha = self.alpha.grad((s,)).into_dense().insert_axis(Axis(1));
        let grad_beta = self.beta.grad((s,)).into_dense().insert_axis(Axis(1));

        let beta::Grad {
            alpha: gl_alpha,
            beta: gl_beta,
        } = self.dist(s).score(std::slice::from_ref(a.borrow()));

        stack![Axis(0), gl_alpha * grad_alpha, gl_beta * grad_beta]
    }
}

impl<'s, S, A, B> Policy<&'s S> for Beta<A, B>
where
    A: Function<(&'s S,), Output = f64>,
    B: Function<(&'s S,), Output = f64>,
{
    type Action = f64;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, s: &'s S) -> f64 { self.dist(s).sample(rng) }

    fn mode(&self, s: &'s S) -> f64 {
        let d = self.dist(s);
        let modes = d.modes();

        if modes.len() == 0 {
            d.mean()
        } else {
            modes[0]
        }
    }
}

impl<'s, A, B, S, U> Handler<StateActionUpdate<&'s S, U>> for Beta<A, B>
where
    U: std::borrow::Borrow<f64>,

    A: Function<(&'s S,), Output = f64> + Handler<StateUpdate<&'s S, f64>>,
    B: Function<(&'s S,), Output = f64> + Handler<StateUpdate<&'s S, f64>>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: StateActionUpdate<&'s S, U>) -> Result<(), ()> {
        let beta::Grad {
            alpha: gl_alpha,
            beta: gl_beta,
        } = self.dist(msg.state).score(std::slice::from_ref(msg.action.borrow()));

        self.alpha
            .handle(StateUpdate {
                state: msg.state,
                error: msg.error * gl_alpha,
            })
            .ok();
        self.beta
            .handle(StateUpdate {
                state: msg.state,
                error: msg.error * gl_beta,
            })
            .ok();

        Ok(())
    }
}

impl<A, B, D> Handler<GradientUpdate<ArrayBase<D, Ix2>>> for Beta<A, B>
where
    A: Parameterised + for<'m> Handler<GradientUpdate<ArrayView2<'m, f64>>>,
    B: Parameterised + for<'m> Handler<GradientUpdate<ArrayView2<'m, f64>>>,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: GradientUpdate<ArrayBase<D, Ix2>>) -> Result<(), ()> {
        self.handle(GradientUpdate(&msg.0))
    }
}

impl<'m, A, B, D> Handler<GradientUpdate<&'m ArrayBase<D, Ix2>>> for Beta<A, B>
where
    A: Parameterised + Handler<GradientUpdate<ArrayView2<'m, f64>>>,
    B: Parameterised + Handler<GradientUpdate<ArrayView2<'m, f64>>>,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: GradientUpdate<&'m ArrayBase<D, Ix2>>) -> Result<(), ()> {
        let n_alpha = self.alpha.n_weights();

        self.alpha
            .handle(GradientUpdate(msg.0.slice(s![0..n_alpha, ..])))
            .ok();
        self.beta
            .handle(GradientUpdate(msg.0.slice(s![n_alpha.., ..])))
            .ok();

        Ok(())
    }
}

impl<A, B, D> Handler<ScaledGradientUpdate<ArrayBase<D, Ix2>>> for Beta<A, B>
where
    A: Parameterised, // + for<'m> Handler<ScaledGradientUpdate<ArrayView2<'m, f64>>>
    B: Parameterised, // + for<'m> Handler<ScaledGradientUpdate<ArrayView2<'m, f64>>>
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: ScaledGradientUpdate<ArrayBase<D, Ix2>>) -> Result<(), ()> {
        self.handle(ScaledGradientUpdate {
            alpha: msg.alpha,
            jacobian: &msg.jacobian,
        })
    }
}

impl<'m, A, B, D> Handler<ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>> for Beta<A, B>
where
    A: Parameterised, // + Handler<ScaledGradientUpdate<ArrayView2<'m, f64>>>
    B: Parameterised, // + Handler<ScaledGradientUpdate<ArrayView2<'m, f64>>>
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>) -> Result<(), ()> {
        let n_alpha = self.alpha.n_weights();

        msg.jacobian
            .slice(s![0..n_alpha, ..])
            .scaled_addto(msg.alpha, &mut self.alpha.weights_view_mut());
        msg.jacobian
            .slice(s![n_alpha.., ..])
            .scaled_addto(msg.alpha, &mut self.beta.weights_view_mut());

        // let (jac_alpha, jac_beta) = if msg.jacobian.nrows() > 1 {
        // (msg.jacobian.column(0).insert_axis(Axis(1)),
        // msg.jacobian.column(1).insert_axis(Axis(1)))
        // } else {
        // let n_alpha = self.alpha.n_weights();
        // let n_beta = self.beta.n_weights();

        // (msg.jacobian.slice(s![0..n_alpha, ..]),
        // msg.jacobian.slice(s![n_alpha..(n_beta + n_alpha), ..]))
        // };

        // self.alpha.handle(ScaledGradientUpdate {
        // alpha: msg.alpha,
        // jacobian: jac_alpha,
        // }).ok();
        // self.beta.handle(ScaledGradientUpdate {
        // alpha: msg.alpha,
        // jacobian: jac_beta,
        // }).ok();

        Ok(())
    }
}
