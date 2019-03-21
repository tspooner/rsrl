use crate::consts::{PI_OVER_2, G};
use crate::core::Vector;
use crate::geometry::{
    continuous::Interval,
    discrete::Ordinal,
    product::LinearSpace,
};
use ndarray::{Ix1, NdIndex, Axis, Array2};
use std::f64::consts::PI;
use super::{runge_kutta4, Domain, Observation, Transition};

// Link masses:
const M1: f64 = 1.0;
const M2: f64 = 1.0;

// Link lengths:
const L1: f64 = 1.0;
#[allow(dead_code)]
const L2: f64 = 1.0;

// Link centre of masses:
const LC1: f64 = 0.5;
const LC2: f64 = 0.5;

// Link moment of intertias:
const I1: f64 = 1.0;
const I2: f64 = 1.0;

const DT: f64 = 0.2;

const LIMITS_THETA1: (f64, f64) = (-PI, PI);
const LIMITS_THETA2: (f64, f64) = (-PI, PI);
const LIMITS_DTHETA1: (f64, f64) = (-4.0 * PI, 4.0 * PI);
const LIMITS_DTHETA2: (f64, f64) = (-9.0 * PI, 9.0 * PI);

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
    fn index_unchecked(&self, strides: &Ix1) -> isize { (*self as usize).index_unchecked(strides) }
}

/// Classic double pendulum control domain.
///
/// The acrobot is a 2-link pendulum environment in which only the second joint
/// actuated. The goal is to swing the end-effector to a distance equal to the
/// length of one link above the base.
///
/// See [https://www.math24.net/double-pendulum/](https://www.math24.net/double-pendulum/)
pub struct Acrobot {
    state: Vector,
}

impl Acrobot {
    fn new(theta1: f64, theta2: f64, dtheta1: f64, dtheta2: f64) -> Acrobot {
        Acrobot {
            state: Vector::from_vec(vec![theta1, theta2, dtheta1, dtheta2]),
        }
    }

    fn update_state(&mut self, a: usize) {
        let fx = |y: &Vector| Acrobot::grad(y);
        let state_aug = stack![Axis(0), self.state, array![ALL_ACTIONS[a] as f64]];
        let mut ns = runge_kutta4(&fx, state_aug, array![0.0, DT]);


        ns[StateIndex::THETA1] = wrap!(LIMITS_THETA1.0, ns[StateIndex::THETA1], LIMITS_THETA1.1);
        ns[StateIndex::THETA2] = wrap!(LIMITS_THETA2.0, ns[StateIndex::THETA2], LIMITS_THETA2.1);

        ns[StateIndex::DTHETA1] =
            clip!(LIMITS_DTHETA1.0, ns[StateIndex::DTHETA1], LIMITS_DTHETA1.1);
        ns[StateIndex::DTHETA2] =
            clip!(LIMITS_DTHETA2.0, ns[StateIndex::DTHETA2], LIMITS_DTHETA2.1);

        self.state = ns;
    }

    fn grad(state: &Vector) -> Vector {
        let theta1 = state[StateIndex::THETA1];
        let theta2 = state[StateIndex::THETA2];
        let dtheta1 = state[StateIndex::DTHETA1];
        let dtheta2 = state[StateIndex::DTHETA2];
        let torque = state[state.len()-1];

        let sin_t2 = theta2.sin();
        let cos_t2 = theta2.cos();

        let d1 = M1 * LC1 * LC1 + M2 * (L1 * L1 + LC2 * LC2 + 2.0 * L1 * LC2 * cos_t2) + I1 + I2;
        let d2 = M2 * (LC2 * LC2 + L1 * LC2 * cos_t2) + I2;

        let phi2 = M2 * LC2 * G * (theta1 + theta2 - PI_OVER_2).cos();
        let phi1 = -1.0 * L1 * LC2 * dtheta2 * dtheta2 * sin_t2
            - 2.0 * M2 * L1 * LC2 * dtheta2 * dtheta1 * sin_t2
            + (M1 * LC1 + M2 * L1) * G * (theta1 - PI_OVER_2).cos() + phi2;

        let ddtheta2 = (torque + d2 / d1 * phi1 - M2 * L1 * LC2 * dtheta1 * dtheta1 * sin_t2 - phi2)
            / (M2 * LC2 * LC2 + I2 - d2 * d2 / d1);
        let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;

        Vector::from_vec(vec![dtheta1, dtheta2, ddtheta1, ddtheta2])
    }
}

impl Default for Acrobot {
    fn default() -> Acrobot { Acrobot::new(0.0, 0.0, 0.0, 0.0) }
}

impl Domain for Acrobot {
    type StateSpace = LinearSpace<Interval>;
    type ActionSpace = Ordinal;

    fn emit(&self) -> Observation<Vector<f64>> {
        if self.is_terminal() {
            Observation::Terminal(self.state.clone())
        } else {
            Observation::Full(self.state.clone())
        }
    }

    fn step(&mut self, action: usize) -> Transition<Vector<f64>, usize> {
        let from = self.emit();

        self.update_state(action);
        let to = self.emit();
        let reward = self.reward(&from, &to);

        Transition {
            from,
            action,
            reward,
            to,
        }
    }

    fn is_terminal(&self) -> bool {
        let theta1 = self.state[StateIndex::THETA1];
        let theta2 = self.state[StateIndex::THETA2];

        theta1.cos() + (theta1 + theta2).cos() < -1.0
    }

    fn reward(&self, _: &Observation<Vector<f64>>, to: &Observation<Vector<f64>>) -> f64 {
        match *to {
            Observation::Terminal(_) => REWARD_TERMINAL,
            _ => REWARD_STEP,
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        LinearSpace::empty() + Interval::bounded(LIMITS_THETA1.0, LIMITS_THETA1.1)
            + Interval::bounded(LIMITS_THETA2.0, LIMITS_THETA2.1)
            + Interval::bounded(LIMITS_DTHETA1.0, LIMITS_DTHETA1.1)
            + Interval::bounded(LIMITS_DTHETA2.0, LIMITS_DTHETA2.1)
    }

    fn action_space(&self) -> Ordinal { Ordinal::new(3) }
    
    fn render(&self, ctx: &mut ggez::Context) {
        use ggez::graphics::*;
        fn rectangle(w: f32, h: f32) -> Vec<Point2> {
            vec![Point2::new(-w*0.5, 0.0),
            Point2::new(w*0.5, 0.0),
            Point2::new(w*0.5, h),
            Point2::new(-w*0.5, h),
            Point2::new(-w*0.5, 0.0)]
        }
        /// Convert from meters to pixels
        fn m(x: f32) -> f32 {
            x * 70.0
        }

        clear(ctx);
        let state = &self.state;
        let (theta1, theta2) = (state[0] as f32, state[1] as f32);

        let width = ctx.conf.window_mode.width as f32;
        let height = ctx.conf.window_mode.height as f32;

        
        // set_color(ctx, [0.5, 0.5, 0.5, 1.0].into())?;
        let pos1 = Point2::new(width*0.5, height*0.5);
        let rect1 = Mesh::new_polygon(ctx, DrawMode::Fill, &rectangle(m(0.1), m(1.0))).unwrap();
        draw(ctx, &rect1, pos1, theta1).unwrap();


        let pos2 = Point2::new(pos1.x - m(1.0)*theta1.sin(), pos1.y + m(1.0)*theta1.cos());
        let rect2 = Mesh::new_polygon(ctx, DrawMode::Fill, &rectangle(m(0.1), m(1.0))).unwrap();
        draw(ctx, &rect2, pos2, theta1 + theta2).unwrap();

        present(ctx);
        // We yield the current thread until the next update
        ggez::timer::yield_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::{Domain, Observation};

    #[test]
    fn test_initial_observation() {
        let m = Acrobot::default();

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
    fn test_is_terminal() {
        assert!(!Acrobot::default().is_terminal());
    }
}
