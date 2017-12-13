use geometry::Space;
use geometry::dimensions;
use geometry::dimensions::Dimension;


pub enum Observation<S: Space, A: Space> {
    Full {
        state: S::Repr,
        actions: Vec<A::Repr>,
    },
    Partial {
        state: S::Repr,
        actions: Vec<A::Repr>,
    },
    Terminal(S::Repr),
}

impl<S: Space, A: Space> Observation<S, A> {
    pub fn state(&self) -> &S::Repr {
        use self::Observation::*;

        match self {
            &Full { ref state, .. } |
            &Partial { ref state, .. } |
            &Terminal(ref state) => state,
        }
    }
}


pub struct Transition<S: Space, A: Space> {
    pub from: Observation<S, A>,
    pub action: A::Repr,
    pub reward: f64,
    pub to: Observation<S, A>,
}


pub trait Domain {
    type StateSpace: Space;
    type ActionSpace: Space;

    fn emit(&self) -> Observation<Self::StateSpace, Self::ActionSpace>;
    fn step(&mut self,
            a: <dimensions::Discrete as Dimension>::Value)
            -> Transition<Self::StateSpace, Self::ActionSpace>;

    fn is_terminal(&self) -> bool;
    fn reward(&self,
              from: &Observation<Self::StateSpace, Self::ActionSpace>,
              to: &Observation<Self::StateSpace, Self::ActionSpace>)
              -> f64;

    fn state_space(&self) -> Self::StateSpace;
    fn action_space(&self) -> Self::ActionSpace;
}


mod ode;
use self::ode::*;

mod mountain_car;
pub use self::mountain_car::MountainCar;

mod cart_pole;
pub use self::cart_pole::CartPole;

mod acrobat;
pub use self::acrobat::Acrobat;

mod hiv;
pub use self::hiv::HIVTreatment;

mod grid_world;
mod cliff_walk;
pub use self::cliff_walk::CliffWalk;
