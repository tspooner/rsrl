use super::{Observation, Transition, Domain};

use geometry::{ActionSpace, RegularSpace};
use geometry::dimensions::{Continuous, Discrete};


const X_MIN: f64 = -1.2;
const X_MAX: f64 = 0.5;

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
        MountainCar { state: MountainCar::initial_state() }
    }
}

impl Domain for MountainCar {
    type StateSpace = RegularSpace<Continuous>;
    // type ActionSpace = UnitarySpace<Discrete>;

    fn emit(&self) -> Observation<Self::StateSpace> {
        // TODO: Don't return all actions.
        if self.is_terminal() {
            Observation::Terminal(vec![self.state.0, self.state.1])
        } else {
            Observation::Full(vec![self.state.0, self.state.1])
        }
    }

    fn step(&mut self, a: usize) -> Transition<Self::StateSpace, ActionSpace> {
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

    fn reward(&self, _: &Observation<Self::StateSpace>,
              to: &Observation<Self::StateSpace>) -> f64
    {
        match to {
            &Observation::Terminal(_) => GOAL_REWARD,
            _ => STEP_REWARD,
        }
    }

    fn is_terminal(&self) -> bool {
        self.state.0 >= X_MAX
    }

    fn state_space(&self) -> Self::StateSpace {
        Self::StateSpace::new()
            .push(Continuous::new(-1.2, 0.5))
            .push(Continuous::new(-0.07, 0.07))
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::new(Discrete::new(3))
    }
}


#[cfg(test)]
mod tests {
    use super::MountainCar;
    use domain::{Observation, Domain};

    #[test]
    fn test_initial_observation() {
        let m = MountainCar::default();

        match m.emit() {
            Observation::Full(ovs) => {
                assert_eq!(ovs[0], -0.5);
                assert_eq!(ovs[1], 0.0);
            }
            _ => panic!("Should yield a fully observable state."),
        }
    }

    // TODO: Write remaining tests.
}
