use geometry::{ActionSpace, RegularSpace};
use geometry::dimensions::{Continuous, Discrete};
use super::{Domain, Observation, Transition};

const X_MIN: f64 = -1.2;
const X_MAX: f64 = 0.6;

const V_MIN: f64 = -0.07;
const V_MAX: f64 = 0.07;

const FORCE_G: f64 = -0.0025;
const FORCE_CAR: f64 = 0.001;

const HILL_FREQ: f64 = 3.0;

const REWARD_STEP: f64 = -1.0;
const REWARD_GOAL: f64 = 0.0;

const ALL_ACTIONS: [f64; 3] = [-1.0, 0.0, 1.0];

/// Classic mountain car testing domain.
///
/// This problem involves an under-powered car which must ascend a steep hill. Since gravity is
/// stronger than the car's engine, even at full throttle, the car cannot simply accelerate up the
/// steep slope. The car is situated in a valley and must learn to leverage potential energy by
/// driving up the opposite hill before the car is able to make it to the goal at the top of the
/// rightmost hill.[^1]
///
/// [^1]: See [https://en.wikipedia.org/wiki/Mountain_car_problem](https://en.wikipedia.org/wiki/Mountain_car_problem)
///
/// # Technical details
/// The **state** is represented by a `Vec` with components:
///
/// | Index | Name     | Min   | Max   |
/// | ----- | -------- | ----- | ----- |
/// | 0     | Position | -1.2  | 0.6   |
/// | 1     | Velocity | -0.07 | 0.07  |
///
///
/// # References
/// - Moore, A. W. (1990). Efficient memory-based learning for robot control.
/// - Singh, S. P., & Sutton, R. S. (1996). Reinforcement learning with replacing eligibility
/// traces. Recent Advances in Reinforcement Learning, 123-158.
/// - Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction (Vol. 1, No.
/// 1). Cambridge: MIT press.
pub struct MountainCar {
    x: f64,
    v: f64,
}

impl MountainCar {
    fn new(x: f64, v: f64) -> MountainCar { MountainCar { x: x, v: v } }

    fn dv(x: f64, a: f64) -> f64 { FORCE_CAR * a + FORCE_G * (HILL_FREQ * x).cos() }

    fn update_state(&mut self, a: usize) {
        let a = ALL_ACTIONS[a];

        self.v = clip!(V_MIN, self.v + Self::dv(self.x, a), V_MAX);
        self.x = clip!(X_MIN, self.x + self.v, X_MAX);
    }
}

impl Default for MountainCar {
    fn default() -> MountainCar { MountainCar::new(-0.5, 0.0) }
}

impl Domain for MountainCar {
    type StateSpace = RegularSpace<Continuous>;
    type ActionSpace = ActionSpace;

    fn emit(&self) -> Observation<Vec<f64>, usize> {
        let s = vec![self.x, self.v];

        if self.is_terminal() {
            Observation::Terminal(s)
        } else {
            Observation::Full {
                state: s,
                actions: [0, 1, 2].iter().cloned().collect(),
            }
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

    fn is_terminal(&self) -> bool { self.x >= X_MAX }

    fn reward(
        &self,
        _: &Observation<Vec<f64>, usize>,
        to: &Observation<Vec<f64>, usize>,
    ) -> f64
    {
        match to {
            &Observation::Terminal(_) => REWARD_GOAL,
            _ => REWARD_STEP,
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        RegularSpace::empty() + Continuous::new(X_MIN, X_MAX) + Continuous::new(V_MIN, V_MAX)
    }

    fn action_space(&self) -> ActionSpace { ActionSpace::new(Discrete::new(3)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use domains::{Domain, Observation};

    #[test]
    fn test_initial_observation() {
        let m = MountainCar::default();

        match m.emit() {
            Observation::Full { ref state, .. } => {
                assert_eq!(state[0], -0.5);
                assert_eq!(state[1], 0.0);
            },
            _ => panic!("Should yield a fully observable state."),
        }
    }

    #[test]
    fn test_is_terminal() {
        assert!(!MountainCar::default().is_terminal());
        assert!(!MountainCar::new(-0.5, 0.0).is_terminal());

        assert!(MountainCar::new(X_MAX, -0.05).is_terminal());
        assert!(MountainCar::new(X_MAX, 0.0).is_terminal());
        assert!(MountainCar::new(X_MAX, 0.05).is_terminal());

        assert!(!MountainCar::new(X_MAX - 0.0001 * X_MAX, 0.0).is_terminal());
        assert!(MountainCar::new(X_MAX + 0.0001 * X_MAX, 0.0).is_terminal());
    }

    #[test]
    fn test_reward() {
        let mc = MountainCar::default();

        let s = mc.emit();
        let ns = MountainCar::new(X_MAX, 0.0).emit();

        assert_eq!(mc.reward(&s, &s), REWARD_STEP);
        assert_eq!(mc.reward(&s, &ns), REWARD_GOAL);
    }
}
