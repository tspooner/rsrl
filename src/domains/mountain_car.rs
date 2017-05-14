use super::{Observation, Transition, Domain};

use geometry::{ActionSpace, RegularSpace};
use geometry::dimensions::{Continuous, Discrete};


const X_MIN: f64 = -1.2;
const X_MAX: f64 = 0.6;

const V_MIN: f64 = -0.07;
const V_MAX: f64 = 0.07;

const FORCE_G: f64 = -0.0025;
const FORCE_CAR: f64 = 0.001;

const HILL_FREQ: f64 = 3.0;

const STEP_REWARD: f64 = -1.0;
const GOAL_REWARD: f64 = 0.0;

const ALL_ACTIONS: [f64; 3] = [-1.0, 0.0, 1.0];


pub struct MountainCar {
    state: (f64, f64),
}

impl MountainCar {
    fn new(state: (f64, f64)) -> MountainCar {
        MountainCar {
            state: state
        }
    }

    fn initial_state() -> (f64, f64) {
        (-0.5, 0.0)
    }

    fn dv(x: f64, a: f64) -> f64 {
        FORCE_CAR * a + FORCE_G * (HILL_FREQ * x).cos()
    }

    fn update_state(&mut self, a: usize) {
        let a = ALL_ACTIONS[a];

        let v = clip!(V_MIN, self.state.1 + Self::dv(self.state.0, a), V_MAX);
        let x = clip!(X_MIN, self.state.0 + v, X_MAX);

        self.state = (x, v);
    }
}

impl Default for MountainCar {
    fn default() -> MountainCar {
        MountainCar::new(MountainCar::initial_state())
    }
}

impl Domain for MountainCar {
    type StateSpace = RegularSpace<Continuous>;
    type ActionSpace = ActionSpace;

    fn emit(&self) -> Observation<Self::StateSpace, Self::ActionSpace> {
        if self.is_terminal() {
            Observation::Terminal(vec![self.state.0, self.state.1])
        } else {
            Observation::Full {
                state: vec![self.state.0, self.state.1],
                actions: vec![0, 1, 2]
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
        self.state.0 >= X_MAX
    }

    fn reward(&self,
              _: &Observation<Self::StateSpace, Self::ActionSpace>,
              to: &Observation<Self::StateSpace, Self::ActionSpace>) -> f64
    {
        match to {
            &Observation::Terminal(_) => GOAL_REWARD,
            _ => STEP_REWARD,
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        Self::StateSpace::new()
            .push(Continuous::new(X_MIN, X_MAX))
            .push(Continuous::new(V_MIN, V_MAX))
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::new(Discrete::new(3))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use domain::{Observation, Domain};

    #[test]
    fn test_initial_observation() {
        let m = MountainCar::default();

        match m.emit() {
            Observation::Full { ref state, .. } => {
                assert_eq!(state[0], -0.5);
                assert_eq!(state[1], 0.0);
            }
            _ => panic!("Should yield a fully observable state."),
        }
    }

    #[test]
    fn test_is_terminal() {
        assert!(!MountainCar::default().is_terminal());
        assert!(!MountainCar::new((-0.5, 0.0)).is_terminal());

        assert!(MountainCar::new((X_MAX, -0.05)).is_terminal());
        assert!(MountainCar::new((X_MAX, 0.0)).is_terminal());
        assert!(MountainCar::new((X_MAX, 0.05)).is_terminal());

        assert!(!MountainCar::new((X_MAX-0.0001*X_MAX, 0.0)).is_terminal());
        assert!(MountainCar::new((X_MAX+0.0001*X_MAX, 0.0)).is_terminal());
    }

    #[test]
    fn test_reward() {
        let mc = MountainCar::default();

        let s = mc.emit();
        let ns = MountainCar::new((X_MAX, 0.0)).emit();

        assert_eq!(mc.reward(&s, &s), STEP_REWARD);
        assert_eq!(mc.reward(&s, &ns), GOAL_REWARD);
    }
}
