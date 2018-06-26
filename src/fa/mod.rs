//! Function approximation and value function representation module.
use core::Shared;
use geometry::Vector;

extern crate lfa;
use self::lfa::approximators::{Multi, Simple};
pub use self::lfa::{
    Approximator,
    Parameterised,
    LFA,

    Projector,
    Projection,
    projectors,

    EvaluationResult,
    UpdateResult,
    AdaptResult,
};

#[cfg(test)]
pub(crate) mod mocking;

mod table;
pub use self::table::Table;

pub type SimpleLFA<S, P> = LFA<S, P, Simple>;
pub type MultiLFA<S, P> = LFA<S, P, Multi>;

pub type SharedVFunction<S> = Shared<VFunction<S, Value = f64>>;
pub type SharedQFunction<S> = Shared<QFunction<S, Value = Vector<f64>>>;

/// An interface for state-value functions.
pub trait VFunction<S: ?Sized>: Approximator<S, Value = f64> {
    #[allow(unused_variables)]
    fn evaluate_phi(&self, phi: &Projection) -> f64 { unimplemented!() }

    #[allow(unused_variables)]
    fn update_phi(&mut self, phi: &Projection, update: f64) { unimplemented!() }
}

impl<S: ?Sized, P: Projector<S>> VFunction<S> for SimpleLFA<S, P> {
    fn evaluate_phi(&self, phi: &Projection) -> f64 { self.approximator.evaluate(phi).unwrap() }

    fn update_phi(&mut self, phi: &Projection, update: f64) {
        let _ = self.approximator.update(phi, update);
    }
}

/// An interface for action-value functions.
pub trait QFunction<S: ?Sized>: Approximator<S, Value = Vector<f64>> {
    fn n_actions(&self) -> usize;

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

impl<S: ?Sized, P: Projector<S>> QFunction<S> for MultiLFA<S, P> {
    fn n_actions(&self) -> usize { self.approximator.weights.cols() }

    fn evaluate_action(&self, input: &S, action: usize) -> f64 {
        let p = self.projector.project(input);

        self.evaluate_action_phi(&p, action)
    }

    fn update_action(&mut self, input: &S, action: usize, update: f64) {
        let p = self.projector.project(input);

        self.update_action_phi(&p, action, update);
    }

    fn evaluate_phi(&self, phi: &Projection) -> Vector<f64> {
        self.approximator.evaluate(&phi).unwrap()
    }

    fn evaluate_action_phi(&self, phi: &Projection, action: usize) -> f64 {
        let col = self.approximator.weights.column(action);

        match phi {
            &Projection::Dense(ref dense) => col.dot(&(dense / phi.z())),
            &Projection::Sparse(ref sparse) => sparse.iter().fold(0.0, |acc, idx| acc + col[*idx]),
        }
    }

    fn update_phi(&mut self, phi: &Projection, updates: Vector<f64>) {
        let _ = self.approximator.update(phi, updates);
    }

    fn update_action_phi(&mut self, phi: &Projection, action: usize, update: f64) {
        let mut col = self.approximator.weights.column_mut(action);

        let z = phi.z();
        let scaled_update = update / z;

        match phi {
            &Projection::Dense(ref dense) => col.scaled_add(scaled_update, dense),
            &Projection::Sparse(ref sparse) => for idx in sparse {
                col[*idx] += scaled_update
            },
        }
    }
}
