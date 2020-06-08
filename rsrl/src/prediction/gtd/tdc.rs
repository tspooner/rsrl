use crate::{
    Handler, Function, Differentiable,
    domains::Transition,
    fa::{StateUpdate, GradientUpdate},
    params::{BufferMut, Parameterised},
    prediction::ValuePredictor,
};

#[derive(Debug, Parameterised)]
pub struct TDC<F> {
    #[weights] pub fa_theta: F,
    pub fa_w: F,

    pub gamma: f64,
}

impl<F: Parameterised> TDC<F> {
    pub fn new(
        fa_theta: F,
        fa_w: F,
        gamma: f64,
    ) -> Self {
        TDC {
            fa_theta,
            fa_w,

            gamma,
        }
    }
}

impl<'m, S, A, F> Handler<&'m Transition<S, A>> for TDC<F>
where
    F: Differentiable<(&'m S,), Output = f64>
        + Handler<StateUpdate<&'m S, f64>>
        + Handler<GradientUpdate<<F as Differentiable<(&'m S,)>>::Jacobian>>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, t: &'m Transition<S, A>) -> Result<(), ()> {
        let (s, ns) = t.states();

        let w_s = self.fa_w.evaluate((s,));
        let theta_s = self.fa_theta.evaluate((s,));

        let td_error = if t.terminated() {
            t.reward - theta_s
        } else {
            t.reward + self.gamma * self.fa_theta.evaluate((ns,)) - theta_s
        };

        self.fa_w.handle(StateUpdate {
            state: s,
            error: td_error - w_s,
        }).ok();

        let grad_s = self.fa_theta.grad((s,));
        let grad_ns = self.fa_theta.grad((ns,));

        let grad = grad_s.merge(&grad_ns, |x, y| td_error * x - w_s * y);

        self.fa_theta.handle(GradientUpdate(grad)).ok();

        Ok(())
    }
}

impl<S, F: Function<(S,), Output = f64>> ValuePredictor<S> for TDC<F> {
    fn predict_v(&self, s: S) -> f64 {
        self.fa_theta.evaluate((s,))
    }
}
