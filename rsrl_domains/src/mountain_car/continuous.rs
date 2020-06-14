use crate::{
    spaces::{real::Interval, ProductSpace, Surjection},
    Domain,
    Observation,
    Reward,
};

const X_MIN: f64 = -1.2;
const X_MAX: f64 = 0.6;

const V_MIN: f64 = -0.07;
const V_MAX: f64 = 0.07;

const FORCE_G: f64 = -0.0025;
const FORCE_CAR: f64 = 0.0015;

const HILL_FREQ: f64 = 3.0;

const REWARD_STEP: f64 = -1.0;
const REWARD_GOAL: f64 = 0.0;

const MIN_ACTION: f64 = -1.0;
const MAX_ACTION: f64 = 1.0;

pub struct ContinuousMountainCar {
    x: f64,
    v: f64,

    action_space: Interval,
}

impl ContinuousMountainCar {
    pub fn new(x: f64, v: f64) -> ContinuousMountainCar {
        ContinuousMountainCar {
            x,
            v,
            action_space: Interval::bounded(MIN_ACTION, MAX_ACTION),
        }
    }

    fn dv(x: f64, a: f64) -> f64 { FORCE_CAR * a + FORCE_G * (HILL_FREQ * x).cos() }

    fn update_state(&mut self, a: f64) {
        let a = self.action_space.map_onto(a);

        self.v = clip!(V_MIN, self.v + Self::dv(self.x, a), V_MAX);
        self.x = clip!(X_MIN, self.x + self.v, X_MAX);
    }
}

impl Default for ContinuousMountainCar {
    fn default() -> ContinuousMountainCar { ContinuousMountainCar::new(-0.5, 0.0) }
}

impl Domain for ContinuousMountainCar {
    type StateSpace = ProductSpace<Interval>;
    type ActionSpace = Interval;

    fn emit(&self) -> Observation<Vec<f64>> {
        if self.x >= X_MAX {
            Observation::Terminal(vec![self.x, self.v])
        } else {
            Observation::Full(vec![self.x, self.v])
        }
    }

    fn step(&mut self, action: &f64) -> (Observation<Vec<f64>>, Reward) {
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

    fn action_space(&self) -> Interval { Interval::bounded(MIN_ACTION, MAX_ACTION) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Domain, Observation};

    #[test]
    fn test_initial_observation() {
        let m = ContinuousMountainCar::default();

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
        assert!(!ContinuousMountainCar::default().emit().is_terminal());
        assert!(!ContinuousMountainCar::new(-0.5, 0.0).emit().is_terminal());

        assert!(ContinuousMountainCar::new(X_MAX, -0.05)
            .emit()
            .is_terminal());
        assert!(ContinuousMountainCar::new(X_MAX, 0.0).emit().is_terminal());
        assert!(ContinuousMountainCar::new(X_MAX, 0.05).emit().is_terminal());

        assert!(!ContinuousMountainCar::new(X_MAX - 0.0001 * X_MAX, 0.0)
            .emit()
            .is_terminal());
        assert!(ContinuousMountainCar::new(X_MAX + 0.0001 * X_MAX, 0.0)
            .emit()
            .is_terminal());
    }
}
