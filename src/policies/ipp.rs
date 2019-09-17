use crate::{
    Algorithm,
    fa::Parameterised,
    geometry::{Matrix, MatrixView, MatrixViewMut},
    policies::{DifferentiablePolicy, Policy},
};
use rand::Rng;
use ndarray::Axis;

/// Independent Policy Pair (IPP).
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct IPP<P1, P2>(pub P1, pub P2);

impl<P1, P2> IPP<P1, P2> {
    pub fn new(p1: P1, p2: P2) -> Self { IPP(p1, p2) }
}

impl<P1: Algorithm, P2: Algorithm> Algorithm for IPP<P1, P2> {
    fn handle_terminal(&mut self) {
        self.0.handle_terminal();
        self.1.handle_terminal();
    }
}

impl<S, P1, P2> Policy<S> for IPP<P1, P2>
where
    P1: Policy<S>,
    P2: Policy<S>,
{
    type Action = (P1::Action, P2::Action);

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, s: &S) -> (P1::Action, P2::Action) {
        (self.0.sample(rng, s), self.1.sample(rng, s))
    }

    fn mpa(&self, s: &S) -> (P1::Action, P2::Action) {
        (self.0.mpa(s), self.1.mpa(s))
    }

    fn probability(&self, s: &S, a: &(P1::Action, P2:: Action)) -> f64 {
        self.0.probability(s, &a.0) * self.1.probability(s, &a.1)
    }
}

impl<S, P1, P2> DifferentiablePolicy<S> for IPP<P1, P2>
where
    P1: DifferentiablePolicy<S>,
    P2: DifferentiablePolicy<S>,
{
    fn update(&mut self, input: &S, a: &Self::Action, error: f64) {
        self.0.update(input, &a.0, error);
        self.1.update(input, &a.1, error);
    }

    fn update_grad(&mut self, grad: &MatrixView<f64>) {
        let w_0 = self.0.weights_dim();

        match w_0 {
            [r, c] if r > 0 => {
                let grad_0 = grad.slice(s![0..r, 0..c]);

                self.0.update_grad(&grad_0);
            },
            _ => {},
        }

        match self.1.weights_dim() {
            [r, c] if r > 0 => {
                let grad_1 = grad.slice(s![0..r, w_0[1]..(w_0[1] + c)]);

                self.1.update_grad(&grad_1);
            },
            _ => {},
        }
    }

    fn update_grad_scaled(&mut self, grad: &MatrixView<f64>, factor: f64) {
        let w_0 = self.0.weights_dim();

        match w_0 {
            [r, c] if r > 0 => {
                let grad_0 = grad.slice(s![0..r, 0..c]);

                self.0.update_grad_scaled(&grad_0, factor);
            },
            _ => {},
        }

        match self.1.weights_dim() {
            [r, c] if r > 0 => {
                let grad_1 = grad.slice(s![0..r, w_0[1]..(w_0[1] + c)]);

                self.1.update_grad_scaled(&grad_1, factor);
            },
            _ => {},
        }
    }

    fn grad_log(&self, input: &S, a: &Self::Action) -> Matrix<f64> {
        let mut gl_0 = self.0.grad_log(input, &a.0);
        let nr_0 = gl_0.rows();

        let mut gl_1 = self.1.grad_log(input, &a.1);
        let nr_1 = gl_1.rows();

        fn resize(gl: Matrix<f64>, n_rows: usize) -> Matrix<f64> {
            let gl_rows = gl.rows();

            let mut new_gl = unsafe { Matrix::uninitialized((n_rows, gl.cols())) };

            new_gl.slice_mut(s![0..gl_rows, ..]).assign(&gl);
            new_gl.slice_mut(s![gl_rows.., ..]).fill(0.0);

            new_gl
        }

        if nr_0 > nr_1 {
            gl_1 = resize(gl_1, nr_0);
        } else if nr_0 < nr_1 {
            gl_0 = resize(gl_0, nr_1);
        }

        stack![Axis(1), gl_0, gl_1]
    }
}

impl<P1: Parameterised, P2: Parameterised> Parameterised for IPP<P1, P2> {
    fn weights(&self) -> Matrix<f64> {
        stack![Axis(1), self.0.weights(), self.1.weights()]
    }

    fn weights_view(&self) -> MatrixView<f64> {
        unimplemented!()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }

    fn weights_dim(&self) -> [usize; 2] {
        let d0 = self.0.weights_dim();
        let d1 = self.1.weights_dim();

        [d0[0].max(d1[0]), d0[1] + d1[1]]
    }
}
