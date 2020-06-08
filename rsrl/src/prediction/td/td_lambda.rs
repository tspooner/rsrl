use crate::{
    Handler, Function, Differentiable,
    domains::{Transition, Observation},
    fa::ScaledGradientUpdate,
    prediction::ValuePredictor,
    traces::Trace,
};
use super::Response;

#[derive(Debug, Parameterised)]
pub struct TDLambda<F, T> {
    #[weights] pub fa_theta: F,
    pub trace: T,

    pub gamma: f64,
    pub lambda: f64,
}

impl<F, T> TDLambda<F, T> {
    pub fn new(
        fa_theta: F,
        trace: T,
        gamma: f64,
        lambda: f64,
    ) -> Self {
        TDLambda {
            fa_theta,
            trace,

            gamma,
            lambda,
        }
    }
}

impl<'m, S, A, F, T> Handler<&'m Transition<S, A>> for TDLambda<F, T>
where
    F: Differentiable<(&'m S,), Output = f64>,
    F: for<'j> Handler<ScaledGradientUpdate<&'j <F as Differentiable<(&'m S,)>>::Jacobian>>,
    T: Trace<Buffer = <F as Differentiable<(&'m S,)>>::Jacobian>,
{
    type Response = Response<()>;
    type Error = ();

    fn handle(&mut self, transition: &'m Transition<S, A>) -> Result<Self::Response, Self::Error> {
        let from = transition.from.state();

        let pred = self.fa_theta.evaluate((from,));
        let grad = self.fa_theta.grad((from,));

        self.trace.scaled_update(self.lambda * self.gamma, &grad);

        let td_error = match transition.to {
            Observation::Terminal(_) => {
                let jacobian = self.trace.deref();
                let td_error = transition.reward - pred;

                self.fa_theta.handle(ScaledGradientUpdate {
                    alpha: td_error,
                    jacobian,
                }).ok();
                self.trace.reset();

                td_error
            },
            Observation::Full(ref to) | Observation::Partial(ref to) => {
                let jacobian = self.trace.deref();
                let td_error =
                    transition.reward + self.gamma * self.fa_theta.evaluate((to,)) - pred;

                self.fa_theta.handle(ScaledGradientUpdate {
                    alpha: td_error,
                    jacobian,
                }).ok();

                td_error
            }
        };

        Ok(Response {
            td_error,
            vfunc_response: (),
        })
    }
}

impl<S, F, T> ValuePredictor<S> for TDLambda<F, T>
where
    F: Function<(S,), Output = f64>,
{
    fn predict_v(&self, s: S) -> f64 {
        self.fa_theta.evaluate((s,))
    }
}
