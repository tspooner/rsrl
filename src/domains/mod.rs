//! Learning benchmark domains module.

use geometry::Space;
use geometry::dimensions;
use geometry::dimensions::Dimension;


/// Container class for data associated with a domain observation.
pub enum Observation<S: Space, A: Space> {
    /// Fully observed state of the environment.
    Full {
        /// Current state of the environment.
        state: S::Repr,

        /// Set of available actions.
        actions: Vec<A::Repr>,
    },

    /// Partially observed state of the environment.
    Partial {
        /// Current state of the environment.
        state: S::Repr,

        /// Set of available actions.
        actions: Vec<A::Repr>,
    },

    /// Terminal state of the environment.
    Terminal(S::Repr),
}

impl<S: Space, A: Space> Observation<S, A> {
    /// Helper function returning a reference to the state values for the given observation.
    pub fn state(&self) -> &S::Repr {
        use self::Observation::*;

        match self {
            &Full { ref state, .. } |
            &Partial { ref state, .. } |
            &Terminal(ref state) => state,
        }
    }
}


/// Container class for data associated with a domain transition.
pub struct Transition<S: Space, A: Space> {
    /// State transitioned _from_, `s`.
    pub from: Observation<S, A>,

    /// Action taken to initiate the transition.
    pub action: A::Repr,

    /// Reward obtained from the transition.
    pub reward: f64,

    /// State transitioned _to_, `s'`.
    pub to: Observation<S, A>,
}


/// An interface for constructing reinforcement learning problem domains.
pub trait Domain {
    /// State space representation type class.
    type StateSpace: Space;

    /// Action space representation type class.
    type ActionSpace: Space;

    /// Emit an observation of the current state of the environment.
    fn emit(&self) -> Observation<Self::StateSpace, Self::ActionSpace>;

    /// Transition the environment forward a single step given an action, `a`.
    fn step(&mut self,
            a: <dimensions::Discrete as Dimension>::Value)
            -> Transition<Self::StateSpace, Self::ActionSpace>;

    /// Returns true if the current state is terminal.
    fn is_terminal(&self) -> bool;

    /// Compute the reward associated with a transition from one state to another.
    fn reward(&self,
              from: &Observation<Self::StateSpace, Self::ActionSpace>,
              to: &Observation<Self::StateSpace, Self::ActionSpace>)
              -> f64;

    /// Returns an instance of the state space type class.
    fn state_space(&self) -> Self::StateSpace;

    /// Returns an instance of the action space type class.
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

mod openai;
pub use self::openai::OpenAIGym;
