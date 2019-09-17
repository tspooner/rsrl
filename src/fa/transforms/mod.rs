/// An interface for differentiable transformations.
pub trait Transform<T: ?Sized> {
    type Output;

    /// Return the value of the transform for input `x`.
    fn transform(&self, x: T) -> Self::Output;

    /// Return the gradient of the transform for input `x`.
    fn grad(&self, x: T) -> T;

    /// Return the gradient of the transform for input `x` scaled the output error.
    fn grad_scaled(&self, x: T, error: Self::Output) -> T;
}

macro_rules! impl_variants {
    ($name:ident) => {
        impl Transform<[f64; 2]> for $name {
            type Output = [f64; 2];

            fn transform(&self, x: [f64; 2]) -> [f64; 2] {
                [
                    Transform::<f64>::transform(self, x[0]),
                    Transform::<f64>::transform(self, x[1])
                ]
            }

            fn grad(&self, x: [f64; 2]) -> [f64; 2] {
                [
                    Transform::<f64>::grad(self, x[0]),
                    Transform::<f64>::grad(self, x[1])
                ]
            }

            fn grad_scaled(&self, x: [f64; 2], errors: [f64; 2]) -> [f64; 2] {
                [
                    Transform::<f64>::grad(self, x[0]) * errors[0],
                    Transform::<f64>::grad(self, x[1]) * errors[1]
                ]
            }
        }

        impl Transform<[f64; 3]> for $name {
            type Output = [f64; 3];

            fn transform(&self, x: [f64; 3]) -> [f64; 3] {
                [
                    Transform::<f64>::transform(self, x[0]),
                    Transform::<f64>::transform(self, x[1]),
                    Transform::<f64>::transform(self, x[2])
                ]
            }

            fn grad(&self, x: [f64; 3]) -> [f64; 3] {
                [
                    Transform::<f64>::grad(self, x[0]),
                    Transform::<f64>::grad(self, x[1]),
                    Transform::<f64>::grad(self, x[2])
                ]
            }

            fn grad_scaled(&self, x: [f64; 3], errors: [f64; 3]) -> [f64; 3] {
                [
                    Transform::<f64>::grad(self, x[0]) * errors[0],
                    Transform::<f64>::grad(self, x[1]) * errors[1],
                    Transform::<f64>::grad(self, x[2]) * errors[2]
                ]
            }
        }

        impl Transform<Vec<f64>> for $name {
            type Output = Vec<f64>;

            fn transform(&self, x: Vec<f64>) -> Vec<f64> {
                x.into_iter().map(|v| Transform::<f64>::transform(self, v)).collect()
            }

            fn grad(&self, x: Vec<f64>) -> Vec<f64> {
                x.into_iter().map(|v| Transform::<f64>::grad(self, v)).collect()
            }

            fn grad_scaled(&self, x: Vec<f64>, errors: Vec<f64>) -> Vec<f64> {
                x.into_iter()
                    .zip(errors.into_iter())
                    .map(|(v, e)| Transform::<f64>::grad(self, v) * e)
                    .collect()
            }
        }

        impl Transform<Array1<f64>> for $name {
            type Output = Array1<f64>;

            fn transform(&self, x: Array1<f64>) -> Array1<f64> {
                x.mapv_into(|v| Transform::<f64>::transform(self, v))
            }

            fn grad(&self, x: Array1<f64>) -> Array1<f64> {
                x.mapv_into(|v| Transform::<f64>::grad(self, v))
            }

            fn grad_scaled(&self, x: Array1<f64>, errors: Array1<f64>) -> Array1<f64> {
                self.grad(x) * errors
            }
        }
    };
}

import_all!(tanh);
import_all!(identity);
import_all!(softplus);
import_all!(logistic);
import_all!(exponential);

// TODO: Add implementations of standard activation functions.
