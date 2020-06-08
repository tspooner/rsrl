use super::Table;
use crate::{
    fa::{StateActionUpdate, StateUpdate},
    Differentiable,
    Enumerable,
    Function,
    Handler,
};
use ndarray::{Array, Array1, Array2, Axis, Dimension, Ix1, Ix2};
use std::{borrow::Borrow, ops::AddAssign};

impl<D: Dimension> Table<Array<f64, D>> {
    pub fn dense(weights: Array<f64, D>) -> Self { Table(weights) }

    pub fn zeros(dim: D) -> Self { Table::dense(Array::zeros(dim)) }
}

impl<D: Dimension> From<Array<f64, D>> for Table<Array<f64, D>> {
    fn from(w: Array<f64, D>) -> Table<Array<f64, D>> { Table::dense(w) }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Implement V(s)
///////////////////////////////////////////////////////////////////////////////////////////////////
impl crate::params::Parameterised for Table<Array1<f64>> {
    fn weights_view(&self) -> crate::params::WeightsView { self.0.view().insert_axis(Axis(1)) }

    fn weights_view_mut(&mut self) -> crate::params::WeightsViewMut {
        self.0.view_mut().insert_axis(Axis(1))
    }
}

impl Function<()> for Table<Array1<f64>> {
    type Output = Vec<f64>;

    fn evaluate(&self, _: ()) -> Vec<f64> { self.0.to_vec() }
}

impl<S: Borrow<usize>> Function<(S,)> for Table<Array1<f64>> {
    type Output = f64;

    fn evaluate(&self, (s,): (S,)) -> f64 { self.0[*s.borrow()] }
}

impl Enumerable<()> for Table<Array1<f64>> {
    fn len(&self, _: ()) -> usize { self.0.len() }
}

impl<S: Borrow<usize>> Differentiable<(S,)> for Table<Array1<f64>> {
    type Jacobian = crate::params::Tile<Ix1, usize>;

    fn grad(&self, (s,): (S,)) -> Self::Jacobian {
        crate::params::Tile::new(self.0.dim(), Some((*s.borrow(), 1.0)))
    }

    fn grad_log(&self, _: (S,)) -> Self::Jacobian { unimplemented!() }
}

impl<S: Borrow<usize>> Handler<StateUpdate<S>> for Table<Array1<f64>> {
    type Response = super::Response;
    type Error = super::Error;

    fn handle(&mut self, msg: StateUpdate<S>) -> Result<Self::Response, Self::Error> {
        self.0[*msg.state.borrow()] += msg.error;

        Ok(super::Response)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Implement Q(s, a)
///////////////////////////////////////////////////////////////////////////////////////////////////
impl crate::params::Parameterised for Table<Array2<f64>> {
    fn weights_view(&self) -> crate::params::WeightsView { self.0.view() }

    fn weights_view_mut(&mut self) -> crate::params::WeightsViewMut { self.0.view_mut() }
}

impl<S: Borrow<usize>> Function<(S,)> for Table<Array2<f64>> {
    type Output = Vec<f64>;

    fn evaluate(&self, (s,): (S,)) -> Vec<f64> { self.0.row(*s.borrow()).to_vec() }
}

impl<S: Borrow<usize>, A: Borrow<usize>> Function<(S, A)> for Table<Array2<f64>> {
    type Output = f64;

    fn evaluate(&self, (s, a): (S, A)) -> f64 { self.0[(*s.borrow(), *a.borrow())] }
}

impl<S: Borrow<usize>> Enumerable<(S,)> for Table<Array2<f64>> {
    fn len(&self, _: (S,)) -> usize { self.0.ncols() }
}

impl<S: Borrow<usize>, A: Borrow<usize>> Differentiable<(S, A)> for Table<Array2<f64>> {
    type Jacobian = crate::params::Tile<Ix2, (usize, usize)>;

    fn grad(&self, (s, a): (S, A)) -> Self::Jacobian {
        crate::params::Tile::new(self.0.dim(), Some(((*s.borrow(), *a.borrow()), 1.0)))
    }

    fn grad_log(&self, _: (S, A)) -> Self::Jacobian { unimplemented!() }
}

impl<S: Borrow<usize>> Handler<StateUpdate<S, Vec<f64>>> for Table<Array2<f64>> {
    type Response = super::Response;
    type Error = super::Error;

    fn handle(&mut self, msg: StateUpdate<S, Vec<f64>>) -> Result<Self::Response, Self::Error> {
        self.0
            .row_mut(*msg.state.borrow())
            .add_assign(&ndarray::aview1(&msg.error));

        Ok(super::Response)
    }
}

impl<S: Borrow<usize>, A: Borrow<usize>> Handler<StateActionUpdate<S, A>> for Table<Array2<f64>> {
    type Response = super::Response;
    type Error = super::Error;

    fn handle(&mut self, msg: StateActionUpdate<S, A>) -> Result<Self::Response, Self::Error> {
        self.0[(*msg.state.borrow(), *msg.action.borrow())] += msg.error;

        Ok(super::Response)
    }
}
