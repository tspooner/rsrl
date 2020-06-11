use crate::{
    fa::{GradientUpdate, ScaledGradientUpdate, StateActionUpdate},
    params::*,
    policies::{sample_probs_with_rng, Policy},
    utils::argmax_first,
    Differentiable,
    Enumerable,
    Function,
    Handler,
};
use ndarray::{Array2, ArrayBase, Data, Ix2};
use rand::Rng;
use std::{f64, iter::FromIterator};

fn softmax<C: FromIterator<f64>>(values: &[f64], tau: f64, c: f64) -> C {
    let mut z = 0.0;

    let ps: Vec<f64> = values
        .into_iter()
        .map(|v| {
            let v = ((v - c) / tau).exp();
            z += v;

            v
        })
        .collect();

    ps.into_iter().map(|v| (v / z).min(f64::MAX)).collect()
}

fn softmax_stable<C: FromIterator<f64>>(values: &[f64], tau: f64) -> C {
    let max_v = values
        .into_iter()
        .fold(f64::NAN, |acc, &v| f64::max(acc, v));

    softmax(values, tau, max_v)
}

pub type Gibbs<F> = Softmax<F>;

#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Softmax<F> {
    #[weights]
    fa: F,

    pub tau: f64,
}

impl<F> Softmax<F> {
    pub fn new(fa: F, tau: f64) -> Self {
        if tau.abs() < 1e-7 {
            panic!("Tau parameter in Softmax must be non-zero.");
        }

        Softmax { fa, tau }
    }

    pub fn standard(fa: F) -> Self { Self::new(fa, 1.0) }
}

impl<'s, S, F: Function<(&'s S,), Output = Vec<f64>>> Function<(&'s S,)> for Softmax<F> {
    type Output = Vec<f64>;

    fn evaluate(&self, (s,): (&'s S,)) -> Vec<f64> {
        let values = self.fa.evaluate((s,));

        softmax_stable(&values, self.tau)
    }
}

impl<'s, S, A, F> Function<(&'s S, A)> for Softmax<F>
where
    A: std::borrow::Borrow<usize>,
    F: Function<(&'s S, usize), Output = f64>,
{
    type Output = f64;

    fn evaluate(&self, (s, a): (&'s S, A)) -> f64 { self.fa.evaluate((s, *a.borrow())) }
}

impl<'s, S, F> Enumerable<(&'s S,)> for Softmax<F>
where F: Enumerable<(&'s S,), Output = Vec<f64>>
{
    fn evaluate_index(&self, (s,): (&'s S,), index: usize) -> f64 {
        self.fa.evaluate_index((s,), index)
    }
}

impl<'s, S, A, F> Differentiable<(&'s S, A)> for Softmax<F>
where
    A: std::borrow::Borrow<usize>,
    F: Function<(&'s S, usize), Output = f64> + Parameterised,
    F: Enumerable<(&'s S,), Output = Vec<f64>>,
    F: Differentiable<(&'s S, usize)>,
{
    type Jacobian = Array2<f64>;

    fn grad(&self, _: (&'s S, A)) -> Array2<f64> { unimplemented!() }

    fn grad_log(&self, (s, a): (&'s S, A)) -> Array2<f64> {
        let a = *a.borrow();

        // (A x 1)
        let mut scale_factors = self.evaluate((s,));
        scale_factors[a] = scale_factors[a] - 1.0;

        // (N x A)
        let mut jac = Array2::zeros(self.weights_dim());

        for (col, sf) in scale_factors.into_iter().enumerate() {
            jac.scaled_add(-sf, &self.fa.grad((&s, col)).into_dense());
        }

        jac
    }
}

impl<'s, S, F> Policy<&'s S> for Softmax<F>
where
    F: Function<(&'s S, usize), Output = f64> + Parameterised,
    F: Enumerable<(&'s S,), Output = Vec<f64>>,
{
    type Action = usize;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, s: &'s S) -> usize {
        sample_probs_with_rng(rng, &self.evaluate((s,)))
    }

    fn mode(&self, s: &'s S) -> usize { argmax_first(self.evaluate((s,))).0 }
}

impl<'s, S, A, F> Handler<StateActionUpdate<&'s S, A>> for Softmax<F>
where
    A: std::borrow::Borrow<usize>,
    F: Handler<ScaledGradientUpdate<<Self as Differentiable<(&'s S, A)>>::Jacobian>>,
    Self: Differentiable<(&'s S, A)>,
{
    type Response = F::Response;
    type Error = F::Error;

    fn handle(&mut self, msg: StateActionUpdate<&'s S, A>) -> Result<Self::Response, Self::Error> {
        let jac = self.grad_log((msg.state, msg.action));

        self.fa.handle(ScaledGradientUpdate {
            alpha: msg.error,
            jacobian: jac,
        })
    }
}

impl<D, F> Handler<GradientUpdate<ArrayBase<D, Ix2>>> for Softmax<F>
where
    F: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: GradientUpdate<ArrayBase<D, Ix2>>) -> Result<(), ()> {
        self.handle(GradientUpdate(&msg.0))
    }
}

impl<'m, D, F> Handler<GradientUpdate<&'m ArrayBase<D, Ix2>>> for Softmax<F>
where
    F: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: GradientUpdate<&'m ArrayBase<D, Ix2>>) -> Result<(), ()> {
        msg.0.addto(&mut self.fa.weights_view_mut());

        Ok(())
    }
}

impl<F, D> Handler<ScaledGradientUpdate<ArrayBase<D, Ix2>>> for Softmax<F>
where
    F: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: ScaledGradientUpdate<ArrayBase<D, Ix2>>) -> Result<(), ()> {
        self.handle(ScaledGradientUpdate {
            alpha: msg.alpha,
            jacobian: &msg.jacobian,
        })
    }
}

impl<'m, F, D> Handler<ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>> for Softmax<F>
where
    F: Parameterised,
    D: Data<Elem = f64>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, msg: ScaledGradientUpdate<&'m ArrayBase<D, Ix2>>) -> Result<(), ()> {
        msg.jacobian
            .scaled_addto(msg.alpha, &mut self.fa.weights_view_mut());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fa::{
            linear::{
                basis::{Basis, Polynomial},
                optim::SGD,
                LFA,
            },
            mocking::MockQ,
        },
    };
    use rand::thread_rng;
    use std::f64::consts::E;

    #[test]
    #[should_panic]
    fn test_0d() {
        let p = Softmax::new(MockQ::new_shared(None), 1.0);

        p.sample(&mut thread_rng(), &vec![]);
    }

    #[test]
    fn test_1d() {
        let p = Softmax::new(MockQ::new_shared(None), 1.0);
        let mut rng = thread_rng();

        for i in 1..100 {
            assert_eq!(p.sample(&mut rng, &vec![i as f64]), 0);
        }
    }

    // #[test]
    // fn test_2d() {
        // let p = Softmax::new(MockQ::new_shared(None), 1.0);
        // let mut rng = thread_rng();
        // let mut counts = vec![0.0, 0.0];

        // for _ in 0..50000 {
            // counts[p.sample(&mut rng, &vec![0.0, 1.0])] += 1.0;
        // }

        // let means: Vec<f64> = counts.into_iter().map(|v| v / 50000.0).collect();

        // assert!(compare_floats(
            // means,
            // &[1.0 / (1.0 + E), E / (1.0 + E)],
            // 1e-2
        // ));
    // }

    // #[test]
    // fn test_probabilites_1() {
        // let p = Softmax::new(MockQ::new_shared(None), 1.0);

        // assert!(compare_floats(
            // p.evaluate((&vec![0.0, 1.0],)),
            // &[1.0 / (1.0 + E), E / (1.0 + E)],
            // 1e-6
        // ));
        // assert!(compare_floats(
            // p.evaluate((&vec![0.0, 2.0],)),
            // &[1.0 / (1.0 + E * E), E * E / (1.0 + E * E)],
            // 1e-6
        // ));
    // }

    // #[test]
    // fn test_probabilities_2() {
    // let fa = LFA::vector(Polynomial::new(1, 1).with_constant(), SGD(1.0), 3);
    // let mut p = Softmax::standard(fa);

    // p.update((&vec![0.0],) &0, -5.0);
    // p.update((&vec![0.0],) &1, 1.0);
    // p.update((&vec![0.0],) &2, -5.0);

    // let ps = p.evaluate((&vec![0.0],));

    // assert!(ps[0] < ps[1]);
    // assert!(ps[2] < ps[1]);
    // }
}
