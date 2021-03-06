use crate::{
    spaces::{discrete::Ordinal, real::Interval, ProductSpace},
    Domain,
    Observation,
    Reward,
};

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
/// This problem involves an under-powered car which must ascend a steep hill.
/// Since gravity is stronger than the car's engine, even at full throttle, the
/// car cannot simply accelerate up the steep slope. The car is situated in a
/// valley and must learn to leverage potential energy by driving up the
/// opposite hill before the car is able to make it to the goal at the top of
/// the rightmost hill.[^1]
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
/// - Singh, S. P., & Sutton, R. S. (1996). Reinforcement learning with
/// replacing eligibility traces. Recent Advances in Reinforcement Learning,
/// 123-158. - Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An
/// introduction (Vol. 1, No. 1). Cambridge: MIT press.
pub struct MountainCar {
    x: f64,
    v: f64,
}

impl MountainCar {
    pub fn new(x: f64, v: f64) -> MountainCar { MountainCar { x, v } }

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
    type StateSpace = ProductSpace<Interval>;
    type ActionSpace = Ordinal;

    fn emit(&self) -> Observation<Vec<f64>> {
        if self.x >= X_MAX {
            Observation::Terminal(vec![self.x, self.v])
        } else {
            Observation::Full(vec![self.x, self.v])
        }
    }

    fn step(&mut self, action: &usize) -> (Observation<Vec<f64>>, Reward) {
        self.update_state(*action);

        let to = self.emit();
        let reward = if to.is_terminal() {
            REWARD_GOAL
        } else {
            REWARD_STEP
        };

        (to, reward)
    }

    fn state_space(&self) -> Self::StateSpace {
        ProductSpace::empty() + Interval::bounded(X_MIN, X_MAX) + Interval::bounded(V_MIN, V_MAX)
    }

    fn action_space(&self) -> Ordinal { Ordinal::new(3) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Domain, Observation};

    #[test]
    fn test_initial_observation() {
        let m = MountainCar::default();

        match m.emit() {
            Observation::Full(ref state) => {
                assert_eq!(state[0], -0.5);
                assert_eq!(state[1], 0.0);
            },
            _ => panic!("Should yield a fully observable state."),
        }
    }

    #[test]
    fn test_is_terminal() {
        assert!(!MountainCar::default().emit().is_terminal());
        assert!(!MountainCar::new(-0.5, 0.0).emit().is_terminal());

        assert!(MountainCar::new(X_MAX, -0.05).emit().is_terminal());
        assert!(MountainCar::new(X_MAX, 0.0).emit().is_terminal());
        assert!(MountainCar::new(X_MAX, 0.05).emit().is_terminal());

        assert!(!MountainCar::new(X_MAX - 0.0001 * X_MAX, 0.0)
            .emit()
            .is_terminal());
        assert!(MountainCar::new(X_MAX + 0.0001 * X_MAX, 0.0)
            .emit()
            .is_terminal());
    }
}
