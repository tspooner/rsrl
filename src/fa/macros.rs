#![macro_use]

macro_rules! impl_newtype_fa {
    ($type:ident.0 => $output:ty) => {
        impl<F: Approximator<Output = $output> + Parameterised> Parameterised for $type<F> {
            fn weights(&self) -> Matrix<f64> { self.0.weights() }

            fn weights_view(&self) -> MatrixView<f64> { self.0.weights_view() }

            fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { self.0.weights_view_mut() }
        }

        impl<I, F: Approximator<Output = $output> + Embedding<I>> Embedding<I> for $type<F> {
            fn n_features(&self) -> usize { self.0.n_features() }

            fn embed(&self, s: &I) -> Features { self.0.embed(s) }
        }

        impl<F: Approximator<Output = $output>> Approximator for $type<F> {
            type Output = F::Output;

            fn n_outputs(&self) -> usize { self.0.n_outputs() }

            fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
                self.0.evaluate(features)
            }

            fn jacobian(&self, features: &Features) -> Matrix<f64> { self.0.jacobian(features) }

            fn update_grad(
                &mut self,
                grad: &Matrix<f64>,
                update: Self::Output,
            ) -> UpdateResult<()>
            {
                self.0.update_grad(grad, update)
            }

            fn update(&mut self, features: &Features, update: Self::Output) -> UpdateResult<()> {
                self.0.update(features, update)
            }
        }
    };
    ($type:ident.$inner:ident => $output:ty) => {
        impl<F: Approximator<Output = $output> + Parameterised> Parameterised for $type<F> {
            fn weights(&self) -> Matrix<f64> { self.$inner.weights() }

            fn weights_view(&self) -> MatrixView<f64> { self.$inner.weights_view() }

            fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { self.$inner.weights_view_mut() }
        }

        impl<I, F: Approximator<Output = $output> + Embedding<I>> Embedding<I> for $type<F> {
            fn n_features(&self) -> usize { self.$inner.n_features() }

            fn to_features(&self, s: &I) -> Features { self.$inner.to_features(s) }
        }

        impl<F: Approximator<Output = $output>> Approximator for $type<F> {
            type Output = F::Output;

            fn n_outputs(&self) -> usize { self.$inner.n_outputs() }

            fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
                self.$inner.evaluate(features)
            }

            fn update(&mut self, features: &Features, update: Self::Output) -> UpdateResult<()> {
                self.$inner.update(features, update)
            }
        }
    };
}
