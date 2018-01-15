use super::{Function, Parameterised, VFunction, QFunction, Projector, Projection};
use geometry::Space;
use ndarray::{ArrayView, Array2};
use std::marker::PhantomData;


#[derive(Serialize, Deserialize)]
pub struct Linear<S: Space, P: Projector<S>> {
    pub projector: P,
    pub weights: Array2<f64>,

    phantom: PhantomData<S>,
}

impl<S: Space, P: Projector<S>> Linear<S, P> {
    pub fn new(projector: P, n_outputs: usize) -> Self {
        let n_features = projector.size();

        Self {
            projector: projector,
            weights: Array2::<f64>::zeros((n_features, n_outputs)),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P: Projector<S>> Function<S::Repr, f64> for Linear<S, P> {
    fn evaluate(&self, input: &S::Repr) -> f64 {
        // Compute the feature vector phi:
        let phi = self.projector.project(input);

        <Self as VFunction<S>>::evaluate_phi(self, &phi)
    }
}

impl<S: Space, P: Projector<S>> Function<S::Repr, Vec<f64>> for Linear<S, P> {
    fn evaluate(&self, input: &S::Repr) -> Vec<f64> {
        // Compute the feature vector phi:
        let phi = self.projector.project(input);

        // Apply matrix multiplication and return Vec<f64>:
        <Self as QFunction<S>>::evaluate_phi(self, &phi)
    }
}

impl<S: Space, P: Projector<S>> Parameterised<S::Repr, f64> for Linear<S, P> {
    fn update(&mut self, input: &S::Repr, error: f64) {
        // Compute the feature vector phi:
        let phi = self.projector.project(input);

        <Self as VFunction<S>>::update_phi(self, &phi, error);
    }
}

impl<S: Space, P: Projector<S>> Parameterised<S::Repr, Vec<f64>> for Linear<S, P> {
    fn update(&mut self, input: &S::Repr, errors: Vec<f64>) {
        // Compute the feature vector phi:
        let phi = self.projector.project(input);

        <Self as QFunction<S>>::update_phi(self, &phi, errors);
    }
}

impl<S: Space, P: Projector<S>> VFunction<S> for Linear<S, P> {
    fn evaluate_phi(&self, phi: &Projection) -> f64 {
        <Self as QFunction<S>>::evaluate_action_phi(self, phi, 0)
    }

    fn update_phi(&mut self, phi: &Projection, error: f64) {
        <Self as QFunction<S>>::update_action_phi(self, phi, 0, error);
    }
}

impl<S: Space, P: Projector<S>> QFunction<S> for Linear<S, P> {
    fn evaluate_action(&self, input: &S::Repr, action: usize) -> f64 {
        let phi = self.projector.project(input);

        self.evaluate_action_phi(&phi, action)
    }

    fn update_action(&mut self, input: &S::Repr, action: usize, error: f64) {
        let phi = self.projector.project(input);

        self.update_action_phi(&phi, action, error);
    }

    fn evaluate_phi(&self, phi: &Projection) -> Vec<f64> {
        match phi {
            &Projection::Dense(ref dense_phi) => (self.weights.t().dot(&(dense_phi/phi.z()))).into_raw_vec(),
            &Projection::Sparse(ref sparse_phi) =>
                (0..self.weights.cols()).map(|c| {
                    sparse_phi.iter().fold(0.0, |acc, idx| acc + self.weights[(*idx, c)])
                }).collect(),
        }
    }

    fn evaluate_action_phi(&self, phi: &Projection, action: usize) -> f64 {
        let col = self.weights.column(action);

        match phi {
            &Projection::Dense(ref dense_phi) => col.dot(&(dense_phi/phi.z())),
            &Projection::Sparse(ref sparse_phi) =>
                sparse_phi.iter().fold(0.0, |acc, idx| acc + col[*idx]),
        }
    }

    fn update_phi(&mut self, phi: &Projection, errors: Vec<f64>) {
        let z = phi.z();
        let sf = 1.0/z;

        match phi {
            &Projection::Dense(ref dense_phi) => {
                let view = dense_phi.view().into_shape((self.weights.rows(), 1)).unwrap();
                let error_matrix = ArrayView::from_shape((1, self.weights.cols()), errors.as_slice())
                    .unwrap();

                self.weights.scaled_add(sf, &view.dot(&error_matrix))
            },
            &Projection::Sparse(ref sparse_phi) => {
                for c in 0..self.weights.cols() {
                    let mut col = self.weights.column_mut(c);
                    let error = errors[c];

                    for idx in sparse_phi {
                        col[*idx] += error
                    }
                }
            },
        }
    }

    fn update_action_phi(&mut self, phi: &Projection, action: usize, error: f64) {
        let mut col = self.weights.column_mut(action);
        let scaled_error = error/phi.z();

        match phi {
            &Projection::Dense(ref dense_phi) => col.scaled_add(scaled_error, dense_phi),
            &Projection::Sparse(ref sparse_phi) => {
                for idx in sparse_phi {
                    col[*idx] += scaled_error
                }
            },
        }
    }
}


#[cfg(test)]
mod tests {
    extern crate seahash;

    use super::*;
    use std::hash::BuildHasherDefault;
    use fa::projection::TileCoding;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_dense_update_eval() {
        let p = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut f = Linear::new(p.clone(), 1);

        let input = vec![5.0];

        f.update(&input, 50.0);
        let out: f64 = f.evaluate(&input);

        assert!(out > 0.0);
    }

    #[test]
    fn test_dense_eval_phi() {
        let p = TileCoding::new(SHBuilder::default(), 4, 100);
        let f = Linear::new(p.clone(), 1);

        let input = vec![5.0];

        let out: f64 = f.evaluate(&input);
        let out_alt_v: f64 = VFunction::evaluate_phi(&f, &p.project(&input));
        let out_alt_q: Vec<f64> = QFunction::evaluate_phi(&f, &p.project(&input));

        assert_eq!(out, out_alt_v);
        assert_eq!(vec![out], out_alt_q);
    }
}
