//! Temporal-difference learning with gradient correction.
use crate::{
    domains::Transition,
    fa::{GradientUpdate, StateUpdate},
    params::BufferMut,
    Differentiable,
    Handler,
};

#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Response<RQ, RT> {
    pub td_error: f64,

    pub q_response: RQ,
    pub td_response: RT,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub enum Error<EQ, ET> {
    QFuncError(EQ),
    TDEstError(ET),
}

/// Temporal-difference learning with gradient correction.
#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct TDC<F> {
    /// Q-function approximator.
    #[weights] pub q_func: F,

    /// TD-error estimator.
    pub td_est: F,

    /// Discount factor.
    pub gamma: f64,
}

type StateMessage<'m, S> = StateUpdate<&'m S, f64>;
type GradientMessage<'m, S, F> = GradientUpdate<<F as Differentiable<(&'m S,)>>::Jacobian>;

impl<'m, S, A, F> Handler<&'m Transition<S, A>> for TDC<F>
where F: Differentiable<(&'m S,), Output = f64>
        + Handler<StateMessage<'m, S>>
        + Handler<GradientMessage<'m, S, F>>
{
    type Response = Response<
        <F as Handler<GradientMessage<'m, S, F>>>::Response,
        <F as Handler<StateMessage<'m, S>>>::Response
    >;
    type Error = Error<
        <F as Handler<GradientMessage<'m, S, F>>>::Error,
        <F as Handler<StateMessage<'m, S>>>::Error
    >;

    fn handle(&mut self, t: &'m Transition<S, A>) -> Result<Self::Response, Self::Error> {
        let (s, ns) = t.states();

        let w_s = self.td_est.evaluate((s,));
        let theta_s = self.q_func.evaluate((s,));

        let td_error = if t.terminated() {
            t.reward - theta_s
        } else {
            t.reward + self.gamma * self.q_func.evaluate((ns,)) - theta_s
        };

        let res_td = self.td_est
            .handle(StateUpdate {
                state: s,
                error: td_error - w_s,
            })
            .map_err(|e| Error::TDEstError(e))?;

        let grad_s = self.q_func.grad((s,));
        let grad_ns = self.q_func.grad((ns,));

        let grad = grad_s.merge(&grad_ns, |x, y| td_error * x - w_s * y);
        let res_q = self.q_func.handle(GradientUpdate(grad)).map_err(|e| Error::QFuncError(e))?;

        Ok(Response {
            td_error,

            q_response: res_q,
            td_response: res_td,
        })
    }
}
