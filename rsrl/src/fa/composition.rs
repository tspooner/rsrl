use crate::{
    Function, Differentiable, Parameterised, Handler,
    params::BufferMut,
    fa::{transforms, StateUpdate, StateActionUpdate},
};

/// Composition of an FA and a differentiable transform.
#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Composition<F, T> {
    #[weights] pub fa: F,
    pub transform: T,
}

impl<F, T> Composition<F, T> {
    pub fn new(fa: F, transform: T) -> Self {
        Composition { fa, transform, }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Common
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<Args, F, T> Function<Args> for Composition<F, T>
where
    F: Function<Args>,
    T: transforms::Transform<F::Output>,
{
    type Output = T::Output;

    fn evaluate(&self, args: Args) -> T::Output {
        self.transform.transform(self.fa.evaluate(args))
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Implement V(s)
///////////////////////////////////////////////////////////////////////////////////////////////////
// TODO: implement for any fa/transform combination, not just those with output f64.
impl<'s, S, F, T> Differentiable<(&'s S,)> for Composition<F, T>
where
    F: Differentiable<(&'s S,), Output = f64>,
    T: transforms::Transform<F::Output, Output = f64>,

    F::Output: Clone,
{
    type Jacobian = F::Jacobian;

    fn grad(&self, args: (&'s S,)) -> Self::Jacobian {
        let gx = self.fa.evaluate(args);
        let f_gx = self.transform.grad(gx);

        self.fa.grad(args).map_into(|x| f_gx * x)
    }

    fn grad_log(&self, args: (&'s S,)) -> Self::Jacobian {
        let gx = self.fa.evaluate(args);
        let fgx = self.transform.transform(gx.clone());
        let f_gx = self.transform.grad(gx);

        self.fa.grad(args).map_into(|x| f_gx * x / fgx)
    }
}

impl<'s, S, F, T> Handler<StateUpdate<&'s S, f64>> for Composition<F, T>
where
    F: Differentiable<(&'s S,), Output = f64> + Handler<StateUpdate<&'s S, f64>>,
    T: transforms::Transform<f64, Output = f64>,
{
    type Response = F::Response;
    type Error = F::Error;

    fn handle(&mut self, msg: StateUpdate<&'s S, f64>) -> Result<Self::Response, Self::Error> {
        let gx = self.fa.evaluate((msg.state,));
        let f_gx = self.transform.grad(gx);

        self.fa.handle(StateUpdate {
            state: msg.state,
            error: msg.error * f_gx,
        })
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Implement Q(s, a)
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<'s, 'a, S, A, F, T> Differentiable<(&'s S, &'a A)> for Composition<F, T>
where
    F: Differentiable<(&'s S, &'a A), Output = f64>,
    T: transforms::Transform<F::Output, Output = f64>,

    F::Output: Clone,
{
    type Jacobian = F::Jacobian;

    fn grad(&self, args: (&'s S, &'a A)) -> Self::Jacobian {
        let gx = self.fa.evaluate(args);
        let f_gx = self.transform.grad(gx);

        self.fa.grad(args).map_into(|x| f_gx * x)
    }

    fn grad_log(&self, args: (&'s S, &'a A)) -> Self::Jacobian {
        let gx = self.fa.evaluate(args);
        let fgx = self.transform.transform(gx.clone());
        let f_gx = self.transform.grad(gx);

        self.fa.grad(args).map_into(|x| f_gx * x / fgx)
    }
}

impl<'s, S, A, F, T> Handler<StateActionUpdate<&'s S, A, f64>> for Composition<F, T>
where
    A: std::borrow::Borrow<f64>,
    F: Differentiable<(&'s S, f64), Output = f64> + Handler<StateActionUpdate<&'s S, A, f64>>,
    T: transforms::Transform<f64, Output = f64>,
{
    type Response = F::Response;
    type Error = F::Error;

    fn handle(&mut self, msg: StateActionUpdate<&'s S, A, f64>) -> Result<Self::Response, Self::Error> {
        let gx = self.fa.evaluate((msg.state, *msg.action.borrow()));
        let f_gx = self.transform.grad(gx);

        self.fa.handle(StateActionUpdate {
            state: msg.state,
            action: msg.action,
            error: msg.error * f_gx,
        })
    }
}
