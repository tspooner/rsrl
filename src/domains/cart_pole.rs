use consts::{FOUR_THIRDS, G, TWELVE_DEGREES};
use core::Vector;
use geometry::{
    dimensions::{Continuous, Discrete},
    RegularSpace,
};
use ndarray::{Ix1, NdIndex};
use super::{runge_kutta4, Domain, Observation, Transition};

const TAU: f64 = 0.02;

const CART_MASS: f64 = 1.0;
const CART_FORCE: f64 = 10.0;

const POLE_COM: f64 = 0.5;
const POLE_MASS: f64 = 0.1;
const POLE_MOMENT: f64 = POLE_COM * POLE_MASS;

const TOTAL_MASS: f64 = CART_MASS + POLE_MASS;

const LIMITS_X: (f64, f64) = (-2.4, 2.4);
const LIMITS_DX: (f64, f64) = (-6.0, 6.0);
const LIMITS_THETA: (f64, f64) = (-TWELVE_DEGREES, TWELVE_DEGREES);
const LIMITS_DTHETA: (f64, f64) = (-2.0, 2.0);

const REWARD_STEP: f64 = 0.0;
const REWARD_TERMINAL: f64 = -1.0;

const ALL_ACTIONS: [f64; 2] = [-1.0 * CART_FORCE, 1.0 * CART_FORCE];

#[derive(Debug, Clone, Copy)]
enum StateIndex {
    X = 0,
    DX = 1,
    THETA = 2,
    DTHETA = 3,
}

unsafe impl NdIndex<Ix1> for StateIndex {
    #[inline]
    fn index_checked(&self, dim: &Ix1, strides: &Ix1) -> Option<isize> {
        (*self as usize).index_checked(dim, strides)
    }

    #[inline(always)]
    fn index_unchecked(&self, strides: &Ix1) -> isize { (*self as usize).index_unchecked(strides) }
}

pub struct CartPole {
    state: Vector,
}

impl CartPole {
    fn new(x: f64, dx: f64, theta: f64, dtheta: f64) -> CartPole {
        CartPole {
            state: Vector::from_vec(vec![x, dx, theta, dtheta]),
        }
    }

    fn update_state(&mut self, a: usize) {
        let fx = |_x, y| CartPole::grad(ALL_ACTIONS[a], y);
        let mut ns = runge_kutta4(&fx, 0.0, self.state.clone(), TAU);

        ns[StateIndex::X] = clip!(LIMITS_X.0, ns[StateIndex::X], LIMITS_X.1);
        ns[StateIndex::DX] = clip!(LIMITS_DX.0, ns[StateIndex::DX], LIMITS_DX.1);

        ns[StateIndex::THETA] = clip!(LIMITS_THETA.0, ns[StateIndex::THETA], LIMITS_THETA.1);
        ns[StateIndex::DTHETA] = clip!(LIMITS_DTHETA.0, ns[StateIndex::DTHETA], LIMITS_DTHETA.1);

        self.state = ns;
    }

    fn grad(force: f64, state: Vector) -> Vector {
        let dx = state[StateIndex::DX];
        let theta = state[StateIndex::THETA];
        let dtheta = state[StateIndex::DTHETA];

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let z = (force + POLE_MOMENT * dtheta * dtheta * sin_theta) / TOTAL_MASS;

        let numer = G * sin_theta - cos_theta * z;
        let denom = FOUR_THIRDS * POLE_COM - POLE_MOMENT * cos_theta * cos_theta;

        let ddtheta = numer / denom;
        let ddx = z - POLE_COM * ddtheta * cos_theta;

        Vector::from_vec(vec![dx, ddx, dtheta, ddtheta])
    }
}

impl Default for CartPole {
    fn default() -> CartPole { CartPole::new(0.0, 0.0, 0.0, 0.0) }
}

impl Domain for CartPole {
    type StateSpace = RegularSpace<Continuous>;
    type ActionSpace = Discrete;

    fn emit(&self) -> Observation<Vec<f64>> {
        if self.is_terminal() {
            Observation::Terminal(self.state.to_vec())
        } else {
            Observation::Full(self.state.to_vec())
        }
    }

    fn step(&mut self, a: usize) -> Transition<Vec<f64>, usize> {
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
        let x = self.state[StateIndex::X];
        let theta = self.state[StateIndex::THETA];

        x <= LIMITS_X.0 || x >= LIMITS_X.1 || theta <= LIMITS_THETA.0 || theta >= LIMITS_THETA.1
    }

    fn reward(&self, _: &Observation<Vec<f64>>, to: &Observation<Vec<f64>>) -> f64 {
        match to {
            &Observation::Terminal(_) => REWARD_TERMINAL,
            _ => REWARD_STEP,
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        RegularSpace::empty() + Continuous::new(LIMITS_X.0, LIMITS_X.1)
            + Continuous::new(LIMITS_DX.0, LIMITS_DX.1)
            + Continuous::new(LIMITS_THETA.0, LIMITS_THETA.1)
            + Continuous::new(LIMITS_DTHETA.0, LIMITS_DTHETA.1)
    }

    fn action_space(&self) -> Discrete { Discrete::new(2) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use domains::{Domain, Observation};

    #[test]
    fn test_initial_observation() {
        let m = CartPole::default();

        match m.emit() {
            Observation::Full(ref state) => {
                assert_eq!(state[0], 0.0);
                assert_eq!(state[1], 0.0);
                assert_eq!(state[2], 0.0);
                assert_eq!(state[3], 0.0);
            },
            _ => panic!("Should yield a fully observable state."),
        }
    }

    #[test]
    fn test_step_0() {
        let mut m = CartPole::default();

        let t = m.step(0);
        let s = t.to.state();
        assert!((s[0] + 0.0032931628891235).abs() < 1e-7);
        assert!((s[1] + 0.3293940797883472).abs() < 1e-7);
        assert!((s[2] - 0.0029499634056967).abs() < 1e-7);
        assert!((s[3] - 0.2951522145037250).abs() < 1e-7);

        let t = m.step(0);
        let s = t.to.state();
        assert!((s[0] + 0.0131819582085161).abs() < 1e-7);
        assert!((s[1] + 0.6597158115002169).abs() < 1e-7);
        assert!((s[2] - 0.0118185373734479).abs() < 1e-7);
        assert!((s[3] - 0.5921703414056713).abs() < 1e-7);
    }

    #[test]
    fn test_step_1() {
        let mut m = CartPole::default();

        let t = m.step(1);
        let s = t.to.state();
        assert!((s[0] - 0.0032931628891235).abs() < 1e-7);
        assert!((s[1] - 0.3293940797883472).abs() < 1e-7);
        assert!((s[2] + 0.0029499634056967).abs() < 1e-7);
        assert!((s[3] + 0.2951522145037250).abs() < 1e-7);

        let t = m.step(1);
        let s = t.to.state();
        assert!((s[0] - 0.0131819582085161).abs() < 1e-7);
        assert!((s[1] - 0.6597158115002169).abs() < 1e-7);
        assert!((s[2] + 0.0118185373734479).abs() < 1e-7);
        assert!((s[3] + 0.5921703414056713).abs() < 1e-7);
    }
}
