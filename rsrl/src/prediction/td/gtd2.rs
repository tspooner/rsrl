use crate::{
    domains::Transition,
    fa::ScaledGradientUpdate,
    params::BufferMut,
    Differentiable,
    Handler,
};

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Response<T, W> {
    pub res_theta: T,
    pub res_w: W,
    pub td_error: f64,
}

#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct GTD2<F> {
    #[weights]
    pub fa_theta: F,
    pub fa_w: F,

    pub gamma: f64,
}

type SGU<'m, S, F> = ScaledGradientUpdate<<F as Differentiable<(&'m S,)>>::Jacobian>;
type SGURef<'m, 'j, S, F> = ScaledGradientUpdate<&'j <F as Differentiable<(&'m S,)>>::Jacobian>;

impl<'m, S, A, F> Handler<&'m Transition<S, A>> for GTD2<F>
where
    F: Differentiable<(&'m S,), Output = f64> + Handler<SGU<'m, S, F>>,
    F: for<'j> Handler<
        SGURef<'m, 'j, S, F>,
        Response = <F as Handler<SGU<'m, S, F>>>::Response,
        Error = <F as Handler<SGU<'m, S, F>>>::Error,
    >,
{
    type Response =
        Response<<F as Handler<SGU<'m, S, F>>>::Response, <F as Handler<SGU<'m, S, F>>>::Response>;
    type Error = <F as Handler<SGU<'m, S, F>>>::Error;

    fn handle(&mut self, t: &'m Transition<S, A>) -> Result<Self::Response, Self::Error> {
        let (s, ns) = t.states();

        let w_s = self.fa_w.evaluate((s,));
        let theta_s = self.fa_theta.evaluate((s,));
        let theta_ns = self.fa_theta.evaluate((ns,));

        let td_error = if t.terminated() {
            t.reward - theta_s
        } else {
            t.reward + self.gamma * theta_ns - theta_s
        };

        let mut grad_s = self.fa_theta.grad((s,));

        let res_w = self.fa_w.handle(ScaledGradientUpdate {
            alpha: td_error - w_s,
            jacobian: &grad_s,
        })?;

        let grad_ns = self.fa_theta.grad((ns,));

        grad_s.merge_inplace(&grad_ns, |x, y| x - self.gamma * y);

        let res_theta = self.fa_theta.handle(ScaledGradientUpdate {
            alpha: w_s,
            jacobian: grad_s,
        })?;

        Ok(Response {
            res_theta,
            res_w,
            td_error,
        })
    }
}
