//! Learning benchmark domains module.
use geometry::Space;
use std::collections::HashSet;

macro_rules! impl_into {
    (Observation<S, $type:ty> => Observation<S, ()>) => (
        impl<S> Into<Observation<S, ()>> for Observation<S, $type> {
            fn into(self) -> Observation<S, ()> {
                use self::Observation::*;

                match self {
                    Full { state, .. } => Full { state: state, actions: HashSet::new() },
                    Partial { state, .. } => Partial { state: state, actions: HashSet::new() },
                    Terminal(state) => Terminal(state),
                }
            }
        }

        impl<'a, S: Clone> From<&'a Observation<S, $type>> for Observation<S, ()> {
            fn from(obs: &'a Observation<S, $type>) -> Observation<S, ()> {
                use self::Observation::*;

                match obs {
                    Full { state, .. } => Full { state: state.clone(), actions: HashSet::new() },
                    Partial { state, .. } => Partial { state: state.clone(), actions: HashSet::new() },
                    Terminal(state) => Terminal(state.clone()),
                }
            }
        }
    );
    (Transition<S, $type:ty> => Transition<S, ()>) => (
        impl<S> Into<Transition<S, ()>> for Transition<S, $type> {
            fn into(self) -> Transition<S, ()> {
                Transition {
                    from: self.from.into(),
                    action: (),
                    reward: self.reward,
                    to: self.to.into(),
                }
            }
        }

        impl<'a, S: Clone> From<&'a Transition<S, $type>> for Transition<S, ()> {
            fn from(t: &'a Transition<S, $type>) -> Transition<S, ()> {
                Transition {
                    from: (&t.from).into(),
                    action: (),
                    reward: t.reward,
                    to: (&t.to).into(),
                }
            }
        }
    )
}

/// Container class for data associated with a domain observation.
#[derive(Clone)]
pub enum Observation<S, A> {
    /// Fully observed state of the environment.
    Full {
        /// Current state of the environment.
        state: S,

        /// Set of available actions.
        actions: HashSet<A>,
    },

    /// Partially observed state of the environment.
    Partial {
        /// Current state of the environment.
        state: S,

        /// Set of available actions.
        actions: HashSet<A>,
    },

    /// Terminal state of the environment.
    Terminal(S),
}

impl<S, A> Observation<S, A> {
    /// Helper function returning a reference to the state values for the given
    /// observation.
    pub fn state(&self) -> &S {
        use self::Observation::*;

        match self {
            &Full { ref state, .. } | &Partial { ref state, .. } | &Terminal(ref state) => state,
        }
    }
}

impl_into!(Observation<S, u8> => Observation<S, ()>);
impl_into!(Observation<S, u16> => Observation<S, ()>);
impl_into!(Observation<S, u32> => Observation<S, ()>);
impl_into!(Observation<S, u64> => Observation<S, ()>);
impl_into!(Observation<S, usize> => Observation<S, ()>);
impl_into!(Observation<S, i8> => Observation<S, ()>);
impl_into!(Observation<S, i16> => Observation<S, ()>);
impl_into!(Observation<S, i32> => Observation<S, ()>);
impl_into!(Observation<S, i64> => Observation<S, ()>);
impl_into!(Observation<S, isize> => Observation<S, ()>);
impl_into!(Observation<S, f32> => Observation<S, ()>);
impl_into!(Observation<S, f64> => Observation<S, ()>);

/// Container class for data associated with a domain transition.
#[derive(Clone)]
pub struct Transition<S, A> {
    /// State transitioned _from_, `s`.
    pub from: Observation<S, A>,

    /// Action taken to initiate the transition (control tasks).
    pub action: A,

    /// Reward obtained from the transition.
    pub reward: f64,

    /// State transitioned _to_, `s'`.
    pub to: Observation<S, A>,
}

impl_into!(Transition<S, u8> => Transition<S, ()>);
impl_into!(Transition<S, u16> => Transition<S, ()>);
impl_into!(Transition<S, u32> => Transition<S, ()>);
impl_into!(Transition<S, u64> => Transition<S, ()>);
impl_into!(Transition<S, usize> => Transition<S, ()>);
impl_into!(Transition<S, i8> => Transition<S, ()>);
impl_into!(Transition<S, i16> => Transition<S, ()>);
impl_into!(Transition<S, i32> => Transition<S, ()>);
impl_into!(Transition<S, i64> => Transition<S, ()>);
impl_into!(Transition<S, isize> => Transition<S, ()>);
impl_into!(Transition<S, f32> => Transition<S, ()>);
impl_into!(Transition<S, f64> => Transition<S, ()>);

/// An interface for constructing reinforcement learning problem domains.
pub trait Domain {
    /// State space representation type class.
    type StateSpace: Space;

    /// Action space representation type class.
    type ActionSpace: Space;

    /// Emit an observation of the current state of the environment.
    fn emit(
        &self,
    ) -> Observation<<Self::StateSpace as Space>::Value, <Self::ActionSpace as Space>::Value>;

    /// Transition the environment forward a single step given an action, `a`.
    fn step(
        &mut self,
        a: <Self::ActionSpace as Space>::Value,
    ) -> Transition<<Self::StateSpace as Space>::Value, <Self::ActionSpace as Space>::Value>;

    /// Returns true if the current state is terminal.
    fn is_terminal(&self) -> bool;

    /// Compute the reward associated with a transition from one state to
    /// another.
    fn reward(
        &self,
        from: &Observation<<Self::StateSpace as Space>::Value, <Self::ActionSpace as Space>::Value>,
        to: &Observation<<Self::StateSpace as Space>::Value, <Self::ActionSpace as Space>::Value>,
    ) -> f64;

    /// Returns an instance of the state space type class.
    fn state_space(&self) -> Self::StateSpace;

    /// Returns an instance of the action space type class.
    fn action_space(&self) -> Self::ActionSpace;
}

mod ode;
use self::ode::*;

mod grid_world;

import_all!(mountain_car);
import_all!(cart_pole);
import_all!(acrobat);
import_all!(hiv);
import_all!(cliff_walk);
import_all!(openai);
