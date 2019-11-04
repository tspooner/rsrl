//! A collection of reinforcement learning benchmark domains.
#[macro_use]
extern crate ndarray;
extern crate spaces;
extern crate rand;

use crate::spaces::Space;
use std::iter;

macro_rules! impl_into {
    (Transition < S, $type:ty > => Transition < S,() >) => {
        impl<S> Into<Transition<S, ()>> for Transition<S, $type> {
            fn into(self) -> Transition<S, ()> { self.drop_action() }
        }
    };
}

macro_rules! make_index {
    ($tname:ident [$($name:ident => $idx:literal),+]) => {
        use std::ops::{Index, IndexMut};

        #[derive(Debug, Clone, Copy)]
        enum $tname {
            $($name = $idx),+
        }

        impl Index<$tname> for Vec<f64> {
            type Output = f64;

            fn index(&self, idx: StateIndex) -> &f64 { self.index(idx as usize) }
        }

        impl Index<$tname> for [f64] {
            type Output = f64;

            fn index(&self, idx: StateIndex) -> &f64 { self.index(idx as usize) }
        }

        impl IndexMut<$tname> for Vec<f64> {
            fn index_mut(&mut self, idx: StateIndex) -> &mut f64 { self.index_mut(idx as usize) }
        }

        impl IndexMut<$tname> for [f64] {
            fn index_mut(&mut self, idx: StateIndex) -> &mut f64 { self.index_mut(idx as usize) }
        }
    }
}

/// Container class for data associated with a domain observation.
#[derive(Clone, Copy, Debug)]
pub enum Observation<S> {
    /// Fully observed state of the environment.
    Full(S),

    /// Partially observed state of the environment.
    Partial(S),

    /// Terminal state of the environment.
    Terminal(S),
}

impl<S> Observation<S> {
    /// Helper function returning a reference to the state values for the given
    /// observation.
    pub fn state(&self) -> &S {
        use self::Observation::*;

        match self {
            Full(ref state) | Partial(ref state) | Terminal(ref state) => state,
        }
    }

    pub fn map<O>(&self, f: impl Fn(&S) -> O) -> Observation<O> {
        use self::Observation::*;

        match self {
            Full(ref state) => Full(f(state)),
            Partial(ref state) => Partial(f(state)),
            Terminal(ref state) => Terminal(f(state)),
        }
    }

    pub fn map_into<O>(&self, f: impl Fn(&S) -> O) -> O {
        use self::Observation::*;

        match self {
            Full(ref state) | Partial(ref state) | Terminal(ref state) => f(state),
        }
    }

    /// Returns true if the state was fully observed, otherwise false.
    pub fn is_full(&self) -> bool {
        match self {
            Observation::Full(_) => true,
            _ => false,
        }
    }

    /// Returns true if the state was only partially observed, otherwise false.
    pub fn is_partial(&self) -> bool {
        match self {
            Observation::Partial(_) => true,
            _ => false,
        }
    }

    /// Returns true if the observation is the terminal state, otherwise false.
    pub fn is_terminal(&self) -> bool {
        match self {
            Observation::Terminal(_) => true,
            _ => false,
        }
    }
}

/// Container class for data associated with a domain transition.
#[derive(Clone, Copy, Debug)]
pub struct Transition<S, A> {
    /// State transitioned _from_, `s`.
    pub from: Observation<S>,

    /// Action taken to initiate the transition (control tasks).
    pub action: A,

    /// Reward obtained from the transition.
    pub reward: f64,

    /// State transitioned _to_, `s'`.
    pub to: Observation<S>,
}

impl<S, A> Transition<S, A> {
    /// Return references to the `from` and `to` states associated with this
    /// transition.
    pub fn states(&self) -> (&S, &S) { (self.from.state(), self.to.state()) }

    /// Apply a closure to the `from` and `to` states associated with this
    /// transition.
    pub fn map_states<O>(self, f: impl Fn(&S) -> O) -> Transition<O, A> {
        Transition {
            from: self.from.map(&f),
            action: self.action,
            reward: self.reward,
            to: self.to.map(f),
        }
    }

    /// Returns true if the transition ends in a terminal state.
    pub fn terminated(&self) -> bool { self.to.is_terminal() }

    /// Replace the action associated with this transition and return a new
    /// instance.
    pub fn replace_action<T>(self, action: T) -> Transition<S, T> {
        Transition {
            from: self.from,
            action: action,
            reward: self.reward,
            to: self.to,
        }
    }

    /// Drop the action associated with this transition and return a new
    /// instance.
    pub fn drop_action(self) -> Transition<S, ()> { self.replace_action(()) }

    pub fn negate_reward(self) -> Transition<S, A> {
        Transition {
            from: self.from,
            action: self.action,
            reward: -self.reward,
            to: self.to,
        }
    }
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

pub type State<D> = <<D as Domain>::StateSpace as Space>::Value;
pub type Action<D> = <<D as Domain>::ActionSpace as Space>::Value;

/// An interface for constructing reinforcement learning problem domains.
pub trait Domain {
    /// State space representation type class.
    type StateSpace: Space;

    /// Action space representation type class.
    type ActionSpace: Space;

    /// Emit an observation of the current state of the environment.
    fn emit(&self) -> Observation<State<Self>>;

    /// Transition the environment forward a single step given an action, `a`.
    fn step(&mut self, a: Action<Self>) -> Transition<State<Self>, Action<Self>>;

    fn rollout(mut self, actor: impl Fn(&State<Self>) -> Action<Self>)
        -> Vec<Transition<State<Self>, Action<Self>>> where Self: Sized
    {
        let first = Some(self.step(self.emit().map_into(&actor)));

        iter::successors(first, |t| match t.to {
            Observation::Terminal(_) => None,
            Observation::Full(ref s) | Observation::Partial(ref s) => Some(self.step(actor(s))),
        }).collect()
    }

    /// Returns an instance of the state space type class.
    fn state_space(&self) -> Self::StateSpace;

    /// Returns an instance of the action space type class.
    fn action_space(&self) -> Self::ActionSpace;
}

mod consts;
mod macros;
mod grid_world;

mod ode;
use self::ode::*;

mod mountain_car;
pub use self::mountain_car::*;

mod cart_pole;
pub use self::cart_pole::*;

mod acrobot;
pub use self::acrobot::*;

mod hiv;
pub use self::hiv::*;

mod cliff_walk;
pub use self::cliff_walk::*;

mod roulette;
pub use self::roulette::*;

#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai")]
pub use self::openai::*;
