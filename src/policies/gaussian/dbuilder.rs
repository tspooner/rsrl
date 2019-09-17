use crate::geometry::{Matrix, Vector};
use rstat::{
    Distribution, ContinuousDistribution,
    univariate::continuous::Normal,
    multivariate::continuous::{BivariateNormal, MultivariateNormal},
};
use std::fmt::Debug;

pub trait DistBuilder<M: Debug + Clone, S: Debug + Clone> {
    type Distribution: ContinuousDistribution;

    fn build(mean: M, stddev: S) -> Self::Distribution;
}

pub struct GB;

#[allow(unused)]
pub(super) type GBSupport<M, S> =
    <<GB as DistBuilder<M, S>>::Distribution as Distribution>::Support;

impl DistBuilder<f64, f64> for GB {
    type Distribution = Normal;

    fn build(mean: f64, stddev: f64) -> Normal {
        Normal::new(mean, stddev)
    }
}

impl DistBuilder<[f64; 2], f64> for GB {
    type Distribution = BivariateNormal;

    fn build(mean: [f64; 2], stddev: f64) -> BivariateNormal {
        BivariateNormal::isotropic(mean, stddev)
    }
}

impl DistBuilder<[f64; 2], [f64; 2]> for GB {
    type Distribution = BivariateNormal;

    fn build(mean: [f64; 2], stddev: [f64; 2]) -> BivariateNormal {
        BivariateNormal::independent(mean, stddev)
    }
}

impl DistBuilder<Vector<f64>, f64> for GB {
    type Distribution = MultivariateNormal;

    fn build(mean: Vector<f64>, stddev: f64) -> MultivariateNormal {
        MultivariateNormal::isotropic(mean, stddev)
    }
}

impl DistBuilder<Vector<f64>, Vector<f64>> for GB {
    type Distribution = MultivariateNormal;

    fn build(mean: Vector<f64>, stddev: Vector<f64>) -> MultivariateNormal {
        let mut sigma = Matrix::eye(mean.len());
        sigma.diag_mut().assign(&stddev);

        MultivariateNormal::new(mean, sigma)
    }
}

impl DistBuilder<Vector<f64>, Matrix<f64>> for GB {
    type Distribution = MultivariateNormal;

    fn build(mean: Vector<f64>, sigma: Matrix<f64>) -> MultivariateNormal {
        MultivariateNormal::new(mean, sigma)
    }
}
