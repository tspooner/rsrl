use crate::{
    Algorithm, OnlineLearner, Parameter,
    domains::Transition,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction, DifferentiableStateFunction,
    },
    linalg::MatrixLike,
    prediction::ValuePredictor,
};

#[derive(Parameterised)]
pub struct GTD2<F> {
    #[weights] pub fa_theta: F,
    pub fa_w: F,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<F: Parameterised> GTD2<F> {
    pub fn new<T1, T2, T3>(
        fa_theta: F,
        fa_w: F,
        alpha: T1,
        beta: T2,
        gamma: T3,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        if fa_theta.weights_dim() != fa_w.weights_dim() {
            panic!("fa_theta and fa_w must be equivalent function approximators.")
        }

        GTD2 {
            fa_theta,
            fa_w,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<F> Algorithm for GTD2<F> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, F> OnlineLearner<S, A> for GTD2<F>
where
    F: DifferentiableStateFunction<S, Output = f64>
{
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let (s, ns) = t.states();

        let w_s = self.fa_w.evaluate(s);
        let theta_s = self.fa_theta.evaluate(s);
        let theta_ns = self.fa_theta.evaluate(ns);

        let td_error = if t.terminated() {
            t.reward - theta_s
        } else {
            t.reward + self.gamma * theta_ns - theta_s
        };

        let grad = self.fa_theta.grad(s);

        self.fa_w.update_grad_scaled(&grad, self.beta * (td_error - w_s));

        let gamma = self.gamma.value();
        let grad = grad.combine(&self.fa_theta.grad(ns), move |x, y| x - gamma * y);

        self.fa_theta.update_grad_scaled(&grad, self.alpha * w_s);
    }
}

impl<S, F> ValuePredictor<S> for GTD2<F>
where
    F: StateFunction<S, Output = f64>
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s)
    }
}
