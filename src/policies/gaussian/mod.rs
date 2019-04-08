use crate::{
    fa::VFunction,
    geometry::Vector,
};

pub(self) struct Mean<F> {
    pub fa: F,
}

impl<F> Mean<F> {
    fn evaluate<S>(&self, input: &S) -> f64
        where F: VFunction<S>,
    {
        self.fa.evaluate(&self.fa.to_features(input)).unwrap()
    }
}

impl<F> Mean<F> {
    fn grad_log<S>(&self, input: &S, a: f64, _std: f64) -> Vector<f64>
        where F: VFunction<S>,
    {
        let phi = self.fa.to_features(input);
        let mean = self.fa.evaluate(&phi).unwrap();
        let phi = phi.expanded(self.fa.n_features());

        (a - mean)/* / std / std*/ * phi
    }
}

import_all!(univariate);
import_all!(bivariate);
import_all!(multivariate);
