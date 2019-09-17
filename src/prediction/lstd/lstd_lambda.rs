use crate::{
    Algorithm, BatchLearner, Parameter,
    domains::Transition,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction,
        linear::LinearStateFunction,
    },
    prediction::ValuePredictor,
    utils::pinv,
};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Solve;

#[derive(Parameterised)]
pub struct LSTDLambda<F> {
    #[weights] pub fa_theta: F,

    pub gamma: Parameter,
    pub lambda: Parameter,

    z: Array1<f64>,

    a: Array2<f64>,
    b: Array1<f64>,
}

impl<F: Parameterised> LSTDLambda<F> {
    pub fn new<T1, T2>(fa_theta: F, gamma: T1, lambda: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        let dim = fa_theta.weights_dim();

        LSTDLambda {
            fa_theta,

            gamma: gamma.into(),
            lambda: lambda.into(),

            z: Array1::zeros(dim[0]),

            a: Array2::eye(dim[0]) * 1e-6,
            b: Array1::zeros(dim[0]),
        }
    }
}

impl<F: Parameterised> LSTDLambda<F> {
    pub fn solve(&mut self) {
        let mut w = self.fa_theta.weights_view_mut();

        if let Ok(theta) = self.a.solve(&self.b) {
            // First try the clean approach:
            w.assign(&theta);
        } else if let Ok(ainv) = pinv(&self.a) {
            // Otherwise solve via SVD:
            w.assign(&ainv.dot(&self.b));
        }
    }
}

impl<F> Algorithm for LSTDLambda<F> {
    fn handle_terminal(&mut self) {
        self.gamma = self.gamma.step();
    }
}

impl<S, A, F> BatchLearner<S, A> for LSTDLambda<F>
where
    F: LinearStateFunction<S, Output = f64>,
{
    fn handle_batch(&mut self, ts: &[Transition<S, A>]) {
        ts.into_iter().for_each(|t| {
            let (s, ns) = t.states();

            let phi_s = self.fa_theta.features(s).expanded();

            // Update trace:
            let decay_rate = self.lambda.value() * self.gamma.value();

            self.z.zip_mut_with(&phi_s, move |x, &y| *x = decay_rate * *x + y);

            // Update matrices:
            self.b.scaled_add(t.reward, &self.z);

            if t.terminated() {
                self.a += &self.z.view().insert_axis(Axis(1)).dot(&phi_s.insert_axis(Axis(0)));
                self.z.fill(0.0);
            } else {
                let mut pd = self.fa_theta.features(ns).expanded();
                let gamma = self.gamma.value();

                pd.zip_mut_with(&phi_s, move |x, &y| *x = y - gamma * *x);

                self.a += &self.z.view().insert_axis(Axis(1)).dot(&pd.insert_axis(Axis(0)));
            }
        });

        self.solve();
    }
}

impl<S, F> ValuePredictor<S> for LSTDLambda<F>
where
    F: StateFunction<S, Output = f64>
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s)
    }
}
