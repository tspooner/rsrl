use crate::{
    core::{Algorithm, Parameter},
    fa::{Approximator, Embedding, Features, Parameterised, VFunction},
    geometry::{continuous::Interval, Matrix, MatrixView, MatrixViewMut, Space, Vector},
    policies::{DifferentiablePolicy, ParameterisedPolicy, Policy},
};
use ndarray::{ArrayBase, Axis, Data, Dimension};
use rand::{rngs::ThreadRng, thread_rng};
use rstat::{univariate::continuous::Normal, ContinuousDistribution, Distribution};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use std::{
    fmt::{self, Debug},
    marker::PhantomData,
    ops::AddAssign,
};

pub mod mean;
use self::mean::Mean;

pub mod stddev;
use self::stddev::StdDev;

import_all!(dbuilder);

#[derive(Clone, Debug, Serialize)]
pub struct Gaussian<M, S> {
    pub mean: M,
    pub stddev: S,

    #[serde(skip_serializing)]
    rng: ThreadRng,
}

impl<M, S> Gaussian<M, S> {
    pub fn new(mean: M, stddev: S) -> Self {
        Gaussian {
            mean,
            stddev,

            rng: thread_rng(),
        }
    }
}

impl<M, S> Gaussian<M, S> {
    #[inline]
    pub fn compute_mean<I>(&self, input: &I) -> M::Output
    where
        M: Mean<I, <S as Approximator>::Output>,
        S: StdDev<I, <M as Approximator>::Output>,
    {
        self.mean.mean(input)
    }

    #[inline]
    pub fn compute_stddev<I>(&self, input: &I) -> S::Output
    where
        M: Mean<I, <S as Approximator>::Output>,
        S: StdDev<I, <M as Approximator>::Output>,
    {
        self.stddev.stddev(input)
    }
}

impl<M, S> Algorithm for Gaussian<M, S> {}

impl<I, M, S> Policy<I> for Gaussian<M, S>
where
    M: Mean<I, <S as Approximator>::Output>,
    M::Output: Clone + Debug,
    S: StdDev<I, <M as Approximator>::Output>,
    S::Output: Clone + Debug,
    GB: DistBuilder<M::Output, S::Output>,
    GBSupport<M::Output, S::Output>: Space<Value = M::Output>,
{
    type Action = M::Output;

    fn sample(&mut self, input: &I) -> Self::Action {
        GB::build(self.compute_mean(input), self.compute_stddev(input)).sample(&mut self.rng)
    }

    fn mpa(&mut self, input: &I) -> Self::Action { self.compute_mean(input) }

    fn probability(&mut self, input: &I, a: Self::Action) -> f64 {
        GB::build(self.compute_mean(input), self.compute_stddev(input)).pdf(a)
    }
}

impl<I, M, S> DifferentiablePolicy<I> for Gaussian<M, S>
where
    M: Mean<I, <S as Approximator>::Output>,
    M::Output: Clone + Debug,
    S: StdDev<I, <M as Approximator>::Output>,
    S::Output: Clone + Debug,
    GB: DistBuilder<M::Output, S::Output>,
    GBSupport<M::Output, S::Output>: Space<Value = M::Output>,
{
    fn grad_log(&self, input: &I, a: Self::Action) -> Matrix<f64> {
        let mean = self.compute_mean(input);
        let stddev = self.compute_stddev(input);

        stack![
            Axis(0),
            self.mean.grad_log(input, &a, stddev),
            self.stddev.grad_log(input, &a, mean)
        ]
    }
}

impl<M, S> Parameterised for Gaussian<M, S>
where
    M: Parameterised,
    S: Parameterised,
{
    fn weights(&self) -> Matrix<f64> { unimplemented!() }

    fn weights_view(&self) -> MatrixView<f64> { unimplemented!() }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { unimplemented!() }

    fn weights_dim(&self) -> (usize, usize) {
        let dim_mean = self.mean.weights_dim();
        let dim_stddev = self.stddev.weights_dim();

        (dim_mean.0 + dim_stddev.0, dim_mean.1)
    }
}

impl<I, M, S> ParameterisedPolicy<I> for Gaussian<M, S>
where
    M: Mean<I, <S as Approximator>::Output> + Parameterised,
    M::Output: Clone + Debug,
    S: StdDev<I, <M as Approximator>::Output> + Parameterised,
    S::Output: Clone + Debug,
    GB: DistBuilder<M::Output, S::Output>,
    GBSupport<M::Output, S::Output>: Space<Value = M::Output>,
{
    fn update(&mut self, input: &I, a: Self::Action, error: f64) {
        let mean = self.compute_mean(input);
        let stddev = self.compute_stddev(input);

        self.mean.update_mean(input, &a, stddev, error);
        self.stddev.update_stddev(input, &a, mean, error);
    }
}

impl<'de, M, S> Deserialize<'de> for Gaussian<M, S>
where
    M: Deserialize<'de>,
    S: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Mean,
            Stddev,
        };

        struct GaussianVisitor<IM, IS>(pub PhantomData<IM>, pub PhantomData<IS>);

        impl<'de, IM, IS> Visitor<'de> for GaussianVisitor<IM, IS>
        where
            IM: Deserialize<'de>,
            IS: Deserialize<'de>,
        {
            type Value = Gaussian<IM, IS>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Gaussian")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Gaussian<IM, IS>, V::Error>
            where V: SeqAccess<'de> {
                let mean = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let stddev = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;

                Ok(Gaussian::new(mean, stddev))
            }

            fn visit_map<V>(self, mut map: V) -> Result<Gaussian<IM, IS>, V::Error>
            where V: MapAccess<'de> {
                let mut mean = None;
                let mut stddev = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Mean => {
                            if mean.is_some() {
                                return Err(de::Error::duplicate_field("mean"));
                            }
                            mean = Some(map.next_value()?);
                        },
                        Field::Stddev => {
                            if stddev.is_some() {
                                return Err(de::Error::duplicate_field("stddev"));
                            }
                            stddev = Some(map.next_value()?);
                        },
                    }
                }

                let mean = mean.ok_or_else(|| de::Error::missing_field("mean"))?;
                let stddev = stddev.ok_or_else(|| de::Error::missing_field("stddev"))?;

                Ok(Gaussian::new(mean, stddev))
            }
        }

        const FIELDS: &'static [&'static str] = &["mean", "stddev"];

        deserializer.deserialize_struct(
            "Gaussian",
            FIELDS,
            GaussianVisitor::<M, S>(PhantomData, PhantomData),
        )
    }
}
