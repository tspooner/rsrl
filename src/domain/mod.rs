use geometry::{Space, ActionSpace};
use geometry::dimensions;
use geometry::dimensions::Dimension;


// TODO: Differentiate between full and partial observations.
pub enum Observation<S: Space> {
    Full(S::Repr),
    Partial(S::Repr),
    Terminal(S::Repr),
}

impl<S: Space> Observation<S> {
    pub fn get(&self) -> &S::Repr {
        use self::Observation::*;

        match self {
            &Full(ref s) | &Partial(ref s) | &Terminal(ref s) => s
        }
    }
}


pub struct Transition<S: Space, A: Space> {
    pub from: Observation<S>,
    pub action: A::Repr,
    pub reward: f64,
    pub to: Observation<S>,
}


pub trait Domain {
    type StateSpace: Space;
    // type ActionSpace: Space;

    fn emit(&self) -> Observation<Self::StateSpace>;
    fn step(&mut self,
            a: <dimensions::Discrete as Dimension>::Value)
            -> Transition<Self::StateSpace, ActionSpace>;

    fn reward(&self,
              from: &Observation<Self::StateSpace>,
              to: &Observation<Self::StateSpace>)
              -> f64;

    fn is_terminal(&self) -> bool;

    fn state_space(&self) -> Self::StateSpace;
    fn action_space(&self) -> ActionSpace;
}


mod mountain_car;
pub use self::mountain_car::MountainCar;
