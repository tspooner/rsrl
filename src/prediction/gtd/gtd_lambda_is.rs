use crate::{
    core::*,
    domains::Transition,
    fa::{VFunction, DifferentiableVFunction, Parameterised},
    geometry::{Space, MatrixView, MatrixViewMut},
    policies::Policy,
};

#[derive(Parameterised)]
pub struct GTDLambdaIS<F, T, B> {
    #[weights] pub fa_theta: F,
    pub fa_w: F,

    trace: Trace,

    target: T,
    behaviour: B,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<F: Parameterised, T, B> GTDLambdaIS<F, T, B> {
    pub fn new<T1, T2, T3>(
        fa_theta: F,
        fa_w: F,
        trace: Trace,
        target: T,
        behaviour: B,
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

        GTDLambdaIS {
            fa_theta,
            fa_w,

            trace,

            target,
            behaviour,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<F, T: Algorithm, B: Algorithm> Algorithm for GTDLambdaIS<F, T, B> {
    fn handle_terminal(&mut self) {
        self.target.handle_terminal();
        self.behaviour.handle_terminal();

        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
        self.trace.lambda = self.trace.lambda.step();

        self.trace.decay(0.0);
    }
}

impl<F, T, B> GTDLambdaIS<F, T, B> {
    fn update_trace<S>(&mut self, t: &Transition<S, T::Action>, phi: &Vector<f64>) -> Features
    where
        F: VFunction<S>,
        T: Policy<S>,
        B: Policy<S, Action = T::Action>,
    {
        let s = t.from.state();

        let decay_rate = self.trace.lambda.value() * self.gamma.value();
        let is_ratio = {
            let pi = self.target.probability(s, &t.action);
            let b = self.behaviour.probability(s, &t.action);

            pi / b
        };

        self.trace.decay(decay_rate);
        self.trace.update(phi);
        self.trace.decay(is_ratio);

        Features::Dense(self.trace.get())
    }
}

impl<S, F, T, B> OnlineLearner<S, T::Action> for GTDLambdaIS<F, T, B>
where
    F: VFunction<S>,
    T: Policy<S>,
    B: Policy<S, Action = T::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, T::Action>) {
        let phi_s = self.fa_theta.embed(t.from.state());

        let v_s = self.fa_theta.evaluate(&phi_s).unwrap();
        let w_s = self.fa_w.evaluate(&phi_s).unwrap();

        let phi_s = phi_s.expanded(self.fa_theta.weights_dim().0);
        let z = self.update_trace(t, &phi_s);
        let phi_s = Features::Dense(phi_s);

        if t.terminated() {
            let residual = t.reward - v_s;

            self.fa_theta.update(&z, self.alpha * residual).ok();
            self.fa_w.update(&z, self.beta * residual).ok();
            self.fa_w.update(&phi_s, self.beta * -w_s).ok();
        } else {
            let phi_ns = self.fa_theta.embed(t.to.state());
            let v_ns = self.fa_theta.evaluate(&phi_ns).unwrap();
            let w_z = self.fa_w.evaluate(&z).unwrap();

            let residual = t.reward + self.gamma * v_ns - v_s;

            self.fa_theta.update(&z, self.alpha * residual).ok();
            self.fa_theta.update(
                &phi_s,
                self.alpha * self.gamma * (self.trace.lambda - 1.0) * w_z
            ).ok();

            self.fa_w.update(&z, self.beta * residual).ok();
            self.fa_w.update(&phi_s, self.beta * -w_s).ok();
        }
    }
}

impl<S, F: VFunction<S>, T, B> ValuePredictor<S> for GTDLambdaIS<F, T, B> {
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(&self.fa_theta.embed(s)).unwrap()
    }
}

impl<S, F, T, B> ActionValuePredictor<S, T::Action> for GTDLambdaIS<F, T, B>
where
    F: VFunction<S>,
    T: Policy<S>,
{}
