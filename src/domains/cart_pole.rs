use super::{Observation, Transition, Domain};

use consts::{TWELVE_DEGREES, FOUR_THIRDS};
use geometry::{ActionSpace, RegularSpace};
use geometry::dimensions::{Continuous, Discrete};


const CART_MASS: f64 = 1.0;
const CART_FORCE: f64 = 10.0;

const POLE_COM: f64 = 0.5;
const POLE_MASS: f64 = 0.1;
const POLE_MOMENT: f64 = POLE_COM * POLE_MASS;

const TOTAL_MASS: f64 = CART_MASS + POLE_MASS;

const G: f64 = 9.8;
const TAU: f64 = 0.02;

const LIMITS_X: (f64, f64) = (-2.4, 2.4);
const LIMITS_X_DOT: (f64, f64) = (-6.0, 6.0);
const LIMITS_THETA: (f64, f64) = (-TWELVE_DEGREES, TWELVE_DEGREES);
const LIMITS_THETA_DOT: (f64, f64) = (-2.0, 2.0);

const REWARD_STEP: f64 = 0.0;
const REWARD_FAIL: f64 = -1.0;

const ALL_ACTIONS: [f64; 2] = [-1.0 * CART_FORCE, 1.0 * CART_FORCE];


pub struct CartPole {
    x: f64,
    x_dot: f64,
    theta: f64,
    theta_dot: f64,
}

impl CartPole {
    fn new(x: f64, x_dot: f64, theta: f64, theta_dot: f64) -> CartPole {
        CartPole {
            x: x,
            x_dot: x_dot,
            theta: theta,
            theta_dot: theta_dot,
        }
    }

    fn update_state(&mut self, a: usize) {
        let a = ALL_ACTIONS[a];

        let cos_theta = self.theta.cos();
        let sin_theta = self.theta.sin();

        let z = (a + POLE_MOMENT * self.theta_dot * self.theta_dot * sin_theta) / TOTAL_MASS;

        let theta_acc = (G * sin_theta - cos_theta * z) /
                        (POLE_COM * (FOUR_THIRDS - POLE_MASS * cos_theta * cos_theta / TOTAL_MASS));

        let x_acc = z - POLE_MOMENT * theta_acc * cos_theta / TOTAL_MASS;

        self.x = clip!(LIMITS_X.0, self.x + TAU * self.x_dot, LIMITS_X.1);
        self.x_dot = clip!(LIMITS_X_DOT.0, self.x_dot + TAU * x_acc, LIMITS_X_DOT.1);

        self.theta = clip!(LIMITS_THETA.0,
                           self.theta + TAU * self.theta_dot,
                           LIMITS_THETA.1);
        self.theta_dot = clip!(LIMITS_THETA_DOT.0,
                               self.theta_dot + TAU * theta_acc,
                               LIMITS_THETA_DOT.1);
    }
}

impl Default for CartPole {
    fn default() -> CartPole {
        CartPole::new(0.0, 0.0, 0.0, 0.0)
    }
}

impl Domain for CartPole {
    type StateSpace = RegularSpace<Continuous>;
    type ActionSpace = ActionSpace;

    fn emit(&self) -> Observation<Self::StateSpace, Self::ActionSpace> {
        let s = vec![self.x, self.x_dot, self.theta, self.theta_dot];

        if self.is_terminal() {
            Observation::Terminal(s)
        } else {
            Observation::Full {
                state: s,
                actions: vec![0, 1],
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
        self.x <= LIMITS_X.0 || self.x >= LIMITS_X.1 || self.theta <= LIMITS_THETA.0 ||
        self.theta >= LIMITS_THETA.1
    }

    fn reward(&self,
              _: &Observation<Self::StateSpace, Self::ActionSpace>,
              to: &Observation<Self::StateSpace, Self::ActionSpace>)
              -> f64 {
        match to {
            &Observation::Terminal(_) => REWARD_FAIL,
            _ => REWARD_STEP,
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        Self::StateSpace::new()
            .push(Continuous::new(LIMITS_X.0, LIMITS_X.1))
            .push(Continuous::new(LIMITS_X_DOT.0, LIMITS_X_DOT.1))
            .push(Continuous::new(LIMITS_THETA.0, LIMITS_THETA.1))
            .push(Continuous::new(LIMITS_THETA_DOT.0, LIMITS_THETA_DOT.1))
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::new(Discrete::new(2))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use domains::{Observation, Domain};

    #[test]
    fn test_initial_observation() {
        let m = CartPole::default();

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
    fn test_step_0() {
        let mut m = CartPole::default();

        let t = m.step(0);
        let s = t.to.state();
        assert!((s[0] - 0.0).abs() < 1e-7);
        assert!((s[1] - -0.1951219512195122).abs() < 1e-7);
        assert!((s[2] - 0.0).abs() < 1e-7);
        assert!((s[3] - 0.2926829268292683).abs() < 1e-7);

        let t = m.step(0);
        let s = t.to.state();
        assert!((s[0] - -0.0039024390243902443).abs() < 1e-7);
        assert!((s[1] - -0.3902439024390244).abs() < 1e-7);
        assert!((s[2] - 0.005853658536585366).abs() < 1e-7);
        assert!((s[3] - 0.5853658536585366).abs() < 1e-7);
    }

    #[test]
    fn test_step_1() {
        let mut m = CartPole::default();

        let t = m.step(1);
        let s = t.to.state();
        assert!((s[0] - 0.0).abs() < 1e-7);
        assert!((s[1] - 0.1951219512195122).abs() < 1e-7);
        assert!((s[2] - 0.0).abs() < 1e-7);
        assert!((s[3] - -0.2926829268292683).abs() < 1e-7);

        let t = m.step(1);
        let s = t.to.state();
        assert!((s[0] - 0.0039024390243902443).abs() < 1e-7);
        assert!((s[1] - 0.3902439024390244).abs() < 1e-7);
        assert!((s[2] - -0.005853658536585366).abs() < 1e-7);
        assert!((s[3] - -0.5853658536585366).abs() < 1e-7);
    }
}
