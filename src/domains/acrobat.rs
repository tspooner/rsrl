use super::{Observation, Transition, Domain, runge_kutta4};

use std::f64::consts::PI;
use consts::{G, PI_OVER_2};
use ndarray::{arr1, Array1, NdIndex, Ix1};
use geometry::{ActionSpace, RegularSpace};
use geometry::dimensions::{Continuous, Discrete};


// Link masses:
const M1: f64 = 1.0;
const M2: f64 = 1.0;

// Link lengths:
const L1: f64 = 1.0;
#[allow(dead_code)] const L2: f64 = 1.0;

// Link centre of masses:
const LC1: f64 = 0.5;
const LC2: f64 = 0.5;

// Link moment of intertias:
const I1: f64 = 1.0;
const I2: f64 = 1.0;

const DT: f64 = 0.2;

const LIMITS_THETA1: (f64, f64) = (-PI, PI);
const LIMITS_THETA2: (f64, f64) = (-PI, PI);
const LIMITS_DTHETA1: (f64, f64) = (-4.0*PI, 4.0*PI);
const LIMITS_DTHETA2: (f64, f64) = (-9.0*PI, 9.0*PI);

const REWARD_STEP: f64 = -1.0;
const REWARD_TERMINAL: f64 = 0.0;

const TORQUE: f64 = 1.0;
const ALL_ACTIONS: [f64; 3] = [-TORQUE, 0.0, TORQUE];


#[derive(Debug, Clone, Copy)]
enum StateIndex {
    THETA1 = 0,
    THETA2 = 1,
    DTHETA1 = 2,
    DTHETA2 = 3,
}

unsafe impl NdIndex<Ix1> for StateIndex {
    #[inline]
    fn index_checked(&self, dim: &Ix1, strides: &Ix1) -> Option<isize> {
        (*self as usize).index_checked(dim, strides)
    }

    #[inline(always)]
    fn index_unchecked(&self, strides: &Ix1) -> isize {
        (*self as usize).index_unchecked(strides)
    }
}


/// Classic double pendulum control domain.
///
/// The acrobot is a 2-link pendulum environment in which only the second joint actuated. The goal
/// is to swing the end-effector to a distance equal to the length of one link above the base.
///
/// See [https://www.math24.net/double-pendulum/](https://www.math24.net/double-pendulum/)
pub struct Acrobat {
    state: Array1<f64>,
}

impl Acrobat {
    fn new(theta1: f64, theta2: f64, dtheta1: f64, dtheta2: f64) -> Acrobat {
        Acrobat {
            state: arr1(&vec![theta1, theta2, dtheta1, dtheta2])
        }
    }

    fn update_state(&mut self, a: usize) {
        let fx = |_x, y| Acrobat::grad(ALL_ACTIONS[a], y);
        let mut ns = runge_kutta4(&fx, 0.0, self.state.clone(), DT);

        ns[StateIndex::THETA1] = wrap!(LIMITS_THETA1.0, ns[StateIndex::THETA1], LIMITS_THETA1.1);
        ns[StateIndex::THETA2] = wrap!(LIMITS_THETA2.0, ns[StateIndex::THETA2], LIMITS_THETA2.1);

        ns[StateIndex::DTHETA1] =
            clip!(LIMITS_DTHETA1.0, ns[StateIndex::DTHETA1], LIMITS_DTHETA1.1);
        ns[StateIndex::DTHETA2] =
            clip!(LIMITS_DTHETA2.0, ns[StateIndex::DTHETA2], LIMITS_DTHETA2.1);

        self.state = ns;
    }

    fn grad(torque: f64, state: Array1<f64>) -> Array1<f64> {
        let theta1 = state[StateIndex::THETA1];
        let theta2 = state[StateIndex::THETA2];
        let dtheta1 = state[StateIndex::DTHETA1];
        let dtheta2 = state[StateIndex::DTHETA2];

        let sin_t2 = theta1.sin();
        let cos_t2 = theta2.cos();

        let d1 =
            M1*LC1*LC1 + M2*(L1*L1 + LC2*LC2 + 2.0*L1*LC2*cos_t2) + I1 + I2;
        let d2 = M2*(LC2*LC2 + L1*LC2*cos_t2) + I2;

        let phi2 = M2*LC2*G*(theta1 + theta2 - PI_OVER_2).cos();
        let phi1 =
            -1.0*L1*LC2*dtheta2*dtheta2*sin_t2 -
            2.0*M2*L1*LC2*dtheta2*dtheta1*sin_t2 +
            (M1*LC1 + M2*L1)*G*(theta1 - PI_OVER_2).cos() + phi2;

        let ddtheta2 =
            (torque + d2/d1*phi1 - M2*L1*LC2*dtheta1*dtheta1*sin_t2 - phi2) /
            (M2*LC2*LC2 + I2 - d2*d2/d1);
        let ddtheta1 = -(d2*ddtheta2 + phi1) / d1;

        arr1(&vec![dtheta1, dtheta2, ddtheta1, ddtheta2])
    }
}

impl Default for Acrobat {
    fn default() -> Acrobat {
        Acrobat::new(0.0, 0.0, 0.0, 0.0)
    }
}

impl Domain for Acrobat {
    type StateSpace = RegularSpace<Continuous>;
    type ActionSpace = ActionSpace;

    fn emit(&self) -> Observation<Self::StateSpace, Self::ActionSpace> {
        if self.is_terminal() {
            Observation::Terminal(self.state.to_vec())
        } else {
            Observation::Full {
                state: self.state.to_vec(),
                actions: vec![0, 1, 2],
            }
        }
    }

    fn step(&mut self, a: usize) -> Transition<Self::StateSpace, Self::ActionSpace> {
        let from = self.emit();

        self.update_state(a);
        let to = self.emit();
        let r = self.reward(&from, &to);

        Transition {
            from: from,
            action: a,
            reward: r,
            to: to,
        }
    }

    fn is_terminal(&self) -> bool {
        let theta1 = self.state[StateIndex::THETA1];
        let theta2 = self.state[StateIndex::THETA2];

        theta1.cos() + (theta1 + theta2).cos() < -1.0
    }

    fn reward(&self,
              _: &Observation<Self::StateSpace, Self::ActionSpace>,
              to: &Observation<Self::StateSpace, Self::ActionSpace>) -> f64
    {
        match to {
            &Observation::Terminal(_) => REWARD_TERMINAL,
            _ => REWARD_STEP,
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        Self::StateSpace::empty()
            .push(Continuous::new(LIMITS_THETA1.0, LIMITS_THETA1.1))
            .push(Continuous::new(LIMITS_THETA2.0, LIMITS_THETA2.1))
            .push(Continuous::new(LIMITS_DTHETA1.0, LIMITS_DTHETA1.1))
            .push(Continuous::new(LIMITS_DTHETA2.0, LIMITS_DTHETA2.1))
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::new(Discrete::new(3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use domains::{Observation, Domain};

    #[test]
    fn test_initial_observation() {
        let m = Acrobat::default();

        match m.emit() {
            Observation::Full { ref state, .. } => {
                assert_eq!(state[0], 0.0);
                assert_eq!(state[1], 0.0);
                assert_eq!(state[2], 0.0);
                assert_eq!(state[3], 0.0);
            }
            _ => panic!("Should yield a fully observable state."),
        }
    }

    #[test]
    fn test_is_terminal() {
        assert!(!Acrobat::default().is_terminal());
    }
}
