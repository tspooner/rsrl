//! Natural actor-critic algorithms.
use crate::{fa::ScaledGradientUpdate, params::*, Handler};

/// Natural actor-critic.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct NAC<C, P> {
    pub critic: C,
    pub policy: P,

    pub alpha: f64,
}

impl<C, P> NAC<C, P> {
    pub fn new(critic: C, policy: P, alpha: f64) -> Self {
        NAC {
            critic,
            policy,

            alpha,
        }
    }
}

impl<M, C, P> Handler<M> for NAC<C, P>
where
    C: Parameterised,
    P: Parameterised
        + Handler<ScaledGradientUpdate<Weights>>
        + for<'w> Handler<
            ScaledGradientUpdate<WeightsView<'w>>,
            Response = <P as Handler<ScaledGradientUpdate<Weights>>>::Response,
            Error = <P as Handler<ScaledGradientUpdate<Weights>>>::Error,
        >,
{
    type Response = <P as Handler<ScaledGradientUpdate<Weights>>>::Response;
    type Error = <P as Handler<ScaledGradientUpdate<Weights>>>::Error;

    fn handle(&mut self, _: M) -> Result<Self::Response, Self::Error> {
        let pw_dim = self.policy.weights_dim();
        let n_features = pw_dim.0 * pw_dim.1;

        let cw = self.critic.weights_view();
        let grad = cw.slice(s![0..n_features, ..]).into_shape(pw_dim).unwrap();
        let norm = grad.fold(0.0, |acc, g| acc + g * g).sqrt().max(1e-3);

        self.policy.handle(ScaledGradientUpdate {
            alpha: self.alpha / norm,
            jacobian: grad,
        })
    }
}
