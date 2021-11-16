//! A collection of reinforcement learning benchmark domains.
#[cfg_attr(test, macro_use)]
extern crate ndarray;
extern crate rand;
extern crate spaces;

use crate::spaces::Space;
use std::iter;

macro_rules! impl_into {
    (Transition < S, $type:ty > => Transition < S,() >) => {
        impl<S> Into<Transition<S, ()>> for Transition<S, $type> {
            fn into(self) -> Transition<S, ()> {
                self.drop_action()
            }
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

pub type Reward = f64;

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

    pub fn borrowed(&self) -> Observation<&S> {
        use self::Observation::*;

        match self {
            Full(ref state) => Full(state),
            Partial(ref state) => Partial(state),
            Terminal(ref state) => Terminal(state),
        }
    }

    /// Returns true if the state was fully observed, otherwise false.
    pub fn is_full(&self) -> bool {
        matches!(self, Observation::Full(_))
    }

    /// Returns true if the state was only partially observed, otherwise false.
    pub fn is_partial(&self) -> bool {
        matches!(self, Observation::Partial(_))
    }

    /// Returns true if the observation is the terminal state, otherwise false.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Observation::Terminal(_))
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
    pub reward: Reward,

    /// State transitioned _to_, `s'`.
    pub to: Observation<S>,
}

impl<S, A> Transition<S, A> {
    /// Return references to the `from` and `to` states associated with this
    /// transition.
    pub fn states(&self) -> (&S, &S) {
        (self.from.state(), self.to.state())
    }

    pub fn borrowed(&self) -> Transition<&S, &A> {
        Transition {
            from: self.from.borrowed(),
            action: &self.action,
            reward: self.reward,
            to: self.to.borrowed(),
        }
    }

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
    pub fn terminated(&self) -> bool {
        self.to.is_terminal()
    }

    /// Replace the action associated with this transition and return a new
    /// instance.
    pub fn replace_action<T>(self, action: T) -> Transition<S, T> {
        Transition {
            from: self.from,
            action,
            reward: self.reward,
            to: self.to,
        }
    }

    /// Drop the action associated with this transition and return a new
    /// instance.
    pub fn drop_action(self) -> Transition<S, ()> {
        self.replace_action(())
    }

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

pub type Batch<S, A> = Vec<Transition<S, A>>;

pub struct TrajectoryIter<'a, S, A> {
    init: &'a Observation<S>,
    tail: &'a [(Observation<S>, A, Reward)],
}

impl<'a, S, A> TrajectoryIter<'a, S, A> {
    #[inline]
    fn next_transition(&self) -> Option<Transition<&'a S, &'a A>> {
        Some(Transition {
            from: self.init.borrowed(),
            action: &self.tail[0].1,
            reward: self.tail[0].2,
            to: self.tail[0].0.borrowed(),
        })
    }
}

impl<'a, S, A> Iterator for TrajectoryIter<'a, S, A> {
    type Item = Transition<&'a S, &'a A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.tail.is_empty() {
            None
        } else {
            let ret = self.next_transition();

            self.init = &self.tail[0].0;
            self.tail = &self.tail[1..];

            ret
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.tail.len();

        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.tail.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n > self.tail.len() - 1 {
            self.tail = &[];

            None
        } else {
            self.init = &self.tail[n - 1].0;
            self.tail = &self.tail[n..];

            self.next_transition()
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        let n = self.tail.len();

        if n == 0 {
            None
        } else if n == 1 {
            self.next_transition()
        } else {
            Some(Transition {
                from: self.tail[n - 2].0.borrowed(),
                action: &self.tail[n - 1].1,
                reward: self.tail[n - 1].2,
                to: self.tail[n - 1].0.borrowed(),
            })
        }
    }
}

impl<'a, S, A> DoubleEndedIterator for TrajectoryIter<'a, S, A> {
    fn next_back(&mut self) -> Option<Transition<&'a S, &'a A>> {
        let n = self.tail.len();

        if n == 0 {
            None
        } else if n == 1 {
            let ret = Some(Transition {
                from: self.init.borrowed(),
                action: &self.tail[0].1,
                reward: self.tail[0].2,
                to: self.tail[0].0.borrowed(),
            });

            self.tail = &self.tail[..(n - 1)];

            ret
        } else {
            let ret = Some(Transition {
                from: self.tail[n - 2].0.borrowed(),
                action: &self.tail[n - 1].1,
                reward: self.tail[n - 1].2,
                to: self.tail[n - 1].0.borrowed(),
            });

            self.tail = &self.tail[..(n - 1)];

            ret
        }
    }
}

impl<'s, 'a, S, A> std::iter::FromIterator<Transition<&'s S, &'a A>> for Vec<Transition<S, A>> {
    fn from_iter<I: IntoIterator<Item = Transition<&'s S, &'a A>>>(iter: I) -> Self {
        iter.into_iter()
            .map(|t| Transition {
                from: t.from,
                action: t.action,
                reward: t.reward,
                to: t.to,
            })
            .collect()
    }
}

pub struct Trajectory<S, A> {
    pub start: Observation<S>,
    pub steps: Vec<(Observation<S>, A, Reward)>,
}

impl<S, A> Trajectory<S, A> {
    pub fn n_states(&self) -> usize {
        self.steps.len() + 1
    }

    pub fn n_transitions(&self) -> usize {
        self.steps.len()
    }

    pub fn first(&self) -> Transition<&S, &A> {
        Transition {
            from: self.start.borrowed(),
            action: &self.steps[0].1,
            reward: self.steps[0].2,
            to: self.steps[0].0.borrowed(),
        }
    }

    pub fn get(&self, index: usize) -> Transition<&S, &A> {
        if index == 0 {
            self.first()
        } else {
            Transition {
                from: self.steps[index].0.borrowed(),
                action: &self.steps[index + 1].1,
                reward: self.steps[index + 1].2,
                to: self.steps[index + 1].0.borrowed(),
            }
        }
    }

    pub fn iter<'a>(&'a self) -> TrajectoryIter<'a, S, A> {
        TrajectoryIter {
            init: &self.start,
            tail: &self.steps,
        }
    }

    pub fn total_reward(&self) -> Reward {
        self.steps.iter().map(|oar| oar.2).sum()
    }

    pub fn to_batch(&self) -> Batch<S, A> {
        self.iter().collect()
    }

    pub fn into_batch(mut self) -> Batch<S, A>
    where
        S: Clone,
        A: Clone,
    {
        if self.n_transitions() == 0 {
            panic!()
        }

        let mut steps = self.steps.drain(..);
        let step_to_first = steps.next().unwrap();

        let mut batch = vec![Transition {
            from: self.start,
            action: step_to_first.1,
            reward: step_to_first.2,
            to: step_to_first.0,
        }];

        for (i, (ns, a, r)) in steps.enumerate() {
            let from = batch[i].from.clone();

            batch.push(Transition {
                from,
                action: a,
                reward: r,
                to: ns,
            });
        }

        batch
    }
}

pub type Trajectories<S, A> = Vec<Trajectory<S, A>>;

pub type State<D> = <<D as Domain>::StateSpace as Space>::Value;
pub type Action<D> = <<D as Domain>::ActionSpace as Space>::Value;

/// An interface for constructing reinforcement learning problem domains.
pub trait Domain {
    /// State space representation type class.
    type StateSpace: Space;

    /// Action space representation type class.
    type ActionSpace: Space;

    /// Returns an instance of the state space type class.
    fn state_space(&self) -> Self::StateSpace;

    /// Returns an instance of the action space type class.
    fn action_space(&self) -> Self::ActionSpace;

    /// Emit an observation of the current state of the environment.
    fn emit(&self) -> Observation<State<Self>>;

    /// Transition the environment forward a single step given an action, `a`.
    fn step(&mut self, a: &Action<Self>) -> (Observation<State<Self>>, Reward);

    fn transition(&mut self, a: Action<Self>) -> Transition<State<Self>, Action<Self>> {
        let s = self.emit();
        let (ns, r) = self.step(&a);

        Transition {
            from: s,
            action: a,
            reward: r,
            to: ns,
        }
    }

    fn rollout<F>(
        mut self,
        mut pi: F,
        step_limit: Option<usize>,
    ) -> Trajectory<State<Self>, Action<Self>>
    where
        F: FnMut(&State<Self>) -> Action<Self>,
        Self: Sized,
    {
        let start = self.emit();
        let action = pi(start.state());
        let step = self.step(&action);

        let iter = iter::successors(Some((step.0, action, step.1)), |(obs, _, _)| match obs {
            Observation::Terminal(_) => None,
            Observation::Full(ref s) | Observation::Partial(ref s) => {
                let a = pi(s);
                let (ns, r) = self.step(&a);

                Some((ns, a, r))
            }
        });

        Trajectory {
            start,
            steps: if let Some(sl) = step_limit {
                iter.take(sl.saturating_sub(1)).collect()
            } else {
                iter.collect()
            },
        }
    }
}

mod consts;
mod grid_world;
mod macros;

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
