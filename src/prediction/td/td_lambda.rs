use crate::{
    Algorithm, OnlineLearner, Parameter,
    domains::Transition,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction, DifferentiableStateFunction,
        traces::Trace,
    },
    prediction::ValuePredictor,
};

#[derive(Parameterised)]
pub struct TDLambda<F, T> {
    #[weights] pub fa_theta: F,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub lambda: Parameter,

    trace: T,
}

impl<F, T> TDLambda<F, T> {
    pub fn new<T1, T2, T3>(
        fa_theta: F,
        trace: T,
        alpha: T1,
        gamma: T2,
        lambda: T3,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        TDLambda {
            fa_theta,

            alpha: alpha.into(),
            gamma: gamma.into(),
            lambda: lambda.into(),

            trace,
        }
    }
}

impl<F, T: Algorithm> Algorithm for TDLambda<F, T> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
        self.lambda = self.lambda.step();

        self.trace.handle_terminal();
    }
}

impl<S, A, F, T> OnlineLearner<S, A> for TDLambda<F, T>
where
    F: DifferentiableStateFunction<S, Output = f64>,
    T: Trace<F::Gradient>,
{
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let s = t.from.state();
        let v = self.fa_theta.evaluate(s);

        self.trace.scaled_update(
            self.lambda.value() * self.gamma.value(),
            &self.fa_theta.grad(s)
        );

        if t.terminated() {
            self.fa_theta.update_grad_scaled(self.trace.deref(), t.reward - v);
            self.trace.reset();
        } else {
            let td_error = t.reward + self.gamma * self.fa_theta.evaluate(t.to.state()) - v;

            self.fa_theta.update_grad_scaled(self.trace.deref(), self.alpha * td_error);
        };
    }
}

impl<S, F, T> ValuePredictor<S> for TDLambda<F, T>
where
    F: StateFunction<S, Output = f64>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s)
    }
}
