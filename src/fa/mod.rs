//! Function approximation and value function representation module.
use crate::{
    core::Shared,
    geometry::Vector,
};

extern crate lfa;
pub use self::lfa::{
    basis::{
        self,
        Projector,
        Projection,
        Composable,
    },
    core::{
        Approximator,
        AdaptResult,
        EvaluationResult,
        UpdateResult,
        Parameterised,
    },
    eval::{ScalarFunction, VectorFunction},
    LFA,
};

#[cfg(test)]
pub(crate) mod mocking;

mod table;
pub use self::table::Table;

pub type ScalarLFA<P> = LFA<P, ScalarFunction>;
pub type VectorLFA<P> = LFA<P, VectorFunction>;

/// An interface for state-value functions.
pub trait VFunction<S: ?Sized>: Approximator<S, Value = f64> {
    #[allow(unused_variables)]
    fn evaluate_phi(&self, phi: &Projection) -> f64 { unimplemented!() }

    #[allow(unused_variables)]
    fn update_phi(&mut self, phi: &Projection, update: f64) { unimplemented!() }
}

impl<S: ?Sized, P: Projector<S>> VFunction<S> for ScalarLFA<P> {
    fn evaluate_phi(&self, phi: &Projection) -> f64 {
        self.evaluator.evaluate(phi).unwrap()
    }

    fn update_phi(&mut self, phi: &Projection, update: f64) {
        self.evaluator.update(phi, update);
    }
}

/// An interface for action-value functions.
pub trait QFunction<S: ?Sized>: Approximator<S, Value = Vector<f64>> {
    fn evaluate_action(&self, input: &S, action: usize) -> f64 {
        self.evaluate(input).unwrap()[action]
    }

    #[allow(unused_variables)]
    fn update_action(&mut self, input: &S, action: usize, update: f64) { unimplemented!() }

    #[allow(unused_variables)]
    fn evaluate_phi(&self, phi: &Projection) -> Vector<f64> { unimplemented!() }

    #[allow(unused_variables)]
    fn evaluate_action_phi(&self, phi: &Projection, action: usize) -> f64 { unimplemented!() }

    #[allow(unused_variables)]
    fn update_phi(&mut self, phi: &Projection, updates: Vector<f64>) { unimplemented!() }

    #[allow(unused_variables)]
    fn update_action_phi(&mut self, phi: &Projection, action: usize, update: f64) {
        unimplemented!()
    }
}

impl<S: ?Sized, P: Projector<S>> QFunction<S> for VectorLFA<P> {
    fn evaluate_action(&self, input: &S, action: usize) -> f64 {
        let p = self.projector.project(input);

        self.evaluate_action_phi(&p, action)
    }

    fn update_action(&mut self, input: &S, action: usize, update: f64) {
        let p = self.projector.project(input);

        self.update_action_phi(&p, action, update);
    }

    fn evaluate_phi(&self, phi: &Projection) -> Vector<f64> {
        self.evaluator.evaluate(&phi).unwrap()
    }

    fn evaluate_action_phi(&self, phi: &Projection, action: usize) -> f64 {
        let col = self.evaluator.weights.column(action);

        match *phi {
            Projection::Dense(ref dense) => col.dot(dense),
            Projection::Sparse(ref sparse) => sparse.iter().fold(0.0, |acc, idx| acc + col[*idx]),
        }
    }

    fn update_phi(&mut self, phi: &Projection, updates: Vector<f64>) {
        let _ = self.evaluator.update(phi, updates);
    }

    fn update_action_phi(&mut self, phi: &Projection, action: usize, update: f64) {
        let mut col = self.evaluator.weights.column_mut(action);

        match *phi {
            Projection::Dense(ref dense) => col.scaled_add(update, dense),
            Projection::Sparse(ref sparse) => {
                for idx in sparse {
                    col[*idx] += update
                }
            },
        }
    }
}
