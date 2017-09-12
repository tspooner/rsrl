use super::{Observation, Transition, Domain};

use std::f64::consts::PI;
use consts::PI_OVER_2;
use geometry::{ActionSpace, RegularSpace};
use geometry::dimensions::{Continuous, Discrete};


const G: f64 = 9.8;

const M1: f64 = 1.0;
const M2: f64 = 1.0;

const L1: f64 = 1.0;
const L2: f64 = 1.0;

const LC1: f64 = 0.5;
const LC2: f64 = 0.5;

const I1: f64 = 1.0;
const I2: f64 = 1.0;

const DT: f64 = 0.05;

const LIMITS_THETA1: (f64, f64) = (-PI, PI);
const LIMITS_THETA2: (f64, f64) = (-PI, PI);
const LIMITS_DTHETA1: (f64, f64) = (-4.0*PI, 4.0*PI);
const LIMITS_DTHETA2: (f64, f64) = (-9.0*PI, 9.0*PI);

const REWARD_STEP: f64 = -1.0;
const REWARD_TERMINAL: f64 = 0.0;

const TORQUE: f64 = 1.0;
const ALL_ACTIONS: [f64; 3] = [-TORQUE, 0.0, TORQUE];


pub struct Acrobat {
    theta1: f64,
    theta2: f64,

    dtheta1: f64,
    dtheta2: f64,
}

impl Acrobat {
    fn new(theta1: f64, theta2: f64, dtheta1: f64, dtheta2: f64) -> Acrobat {
        Acrobat {
            theta1: theta1,
            theta2: theta2,

            dtheta1: dtheta1,
            dtheta2: dtheta2,
        }
    }

    fn update_state(&mut self, a: usize) {
        let torque = ALL_ACTIONS[a];

        let ddtheta = Acrobat::grad(torque, self.theta1, self.theta2,
                                    self.dtheta1, self.dtheta2);
        let ddtheta = (ddtheta.0 * DT / 4.0, ddtheta.1 * DT / 4.0);

        self.theta1 =
            clip!(LIMITS_THETA1.0, self.theta1 + self.dtheta1, LIMITS_THETA1.1);
        self.theta2 =
            clip!(LIMITS_THETA2.0, self.theta2 + self.dtheta2, LIMITS_THETA2.1);

        self.dtheta1 =
            clip!(LIMITS_DTHETA1.0, self.dtheta1 + ddtheta.0, LIMITS_DTHETA1.1);
        self.dtheta2 =
            clip!(LIMITS_DTHETA2.0, self.dtheta2 + ddtheta.1, LIMITS_DTHETA2.1);
    }

    fn grad(torque: f64, theta1: f64, theta2: f64,
            dtheta1: f64, dtheta2: f64) -> (f64, f64)
    {
        let sin_t2 = theta2.sin();
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

        (ddtheta1, ddtheta2)
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
        let s = vec![self.theta1, self.theta2, self.dtheta1, self.dtheta2];

        if self.is_terminal() {
            Observation::Terminal(s)
        } else {
            Observation::Full {
                state: s,
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
        self.theta1.cos() + (self.theta1 + self.theta2).cos() > -1.0
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
        Self::StateSpace::new()
            .push(Continuous::new(LIMITS_THETA1.0, LIMITS_THETA1.1))
            .push(Continuous::new(LIMITS_THETA2.0, LIMITS_THETA2.1))
            .push(Continuous::new(LIMITS_DTHETA1.0, LIMITS_DTHETA1.1))
            .push(Continuous::new(LIMITS_DTHETA2.0, LIMITS_DTHETA2.1))
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::new(Discrete::new(3))
    }
}
