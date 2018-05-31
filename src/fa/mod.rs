//! Function approximation and value function representation module.

use geometry::Vector;

extern crate lfa;
pub use self::lfa::*;

mod table;
pub use self::table::Table;

/// An interface for state-value functions.
pub trait VFunction<I: ?Sized>: Approximator<I, Value = f64> {
    #[allow(unused_variables)]
    fn evaluate_phi(&self, phi: &Projection) -> f64 { unimplemented!() }

    #[allow(unused_variables)]
    fn update_phi(&mut self, phi: &Projection, update: f64) { unimplemented!() }
}

impl<I: ?Sized, P: Projector<I>> VFunction<I> for SimpleLinear<I, P> {
    fn evaluate_phi(&self, phi: &Projection) -> f64 { self.evaluate_projection(phi) }

    fn update_phi(&mut self, phi: &Projection, update: f64) { self.update_projection(phi, update); }
}

/// An interface for action-value functions.
pub trait QFunction<I: ?Sized>: Approximator<I, Value = Vector<f64>> {
    fn evaluate_action(&self, input: &I, action: usize) -> f64 {
        self.evaluate(input).unwrap()[action]
    }

    #[allow(unused_variables)]
    fn update_action(&mut self, input: &I, action: usize, update: f64) { unimplemented!() }

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

impl<I: ?Sized, P: Projector<I>> QFunction<I> for MultiLinear<I, P> {
    fn evaluate_action(&self, input: &I, action: usize) -> f64 {
        let p = self.projector.project(input);

        self.evaluate_action_phi(&p, action)
    }

    fn update_action(&mut self, input: &I, action: usize, update: f64) {
        let p = self.projector.project(input);

        self.update_action_phi(&p, action, update);
    }

    fn evaluate_phi(&self, phi: &Projection) -> Vector<f64> { self.evaluate_projection(&phi) }

    fn evaluate_action_phi(&self, phi: &Projection, action: usize) -> f64 {
        let col = self.weights.column(action);

        match phi {
            &Projection::Dense(ref dense) => col.dot(&(dense / phi.z())),
            &Projection::Sparse(ref sparse) => sparse.iter().fold(0.0, |acc, idx| acc + col[*idx]),
        }
    }

    fn update_phi(&mut self, phi: &Projection, updates: Vector<f64>) {
        self.update_projection(phi, updates);
    }

    fn update_action_phi(&mut self, phi: &Projection, action: usize, update: f64) {
        let mut col = self.weights.column_mut(action);

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
