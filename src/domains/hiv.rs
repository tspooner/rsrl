use super::{Observation, Transition, Domain, runge_kutta4};

use std::ops::Index;
use ndarray::{arr1, Array1, NdIndex, Ix1};
use geometry::{ActionSpace, RegularSpace};
use geometry::dimensions::{Continuous, Discrete};


// Model parameters
// (https://pdfs.semanticscholar.org/c030/127238b1dbad2263fba6b64b5dec7c3ffa20.pdf):
const LAMBDA1: f64 = 1e4;
const LAMBDA2: f64 = 31.98;
const D1: f64 = 0.01;
const D2: f64 = 0.01;
const F: f64 = 0.34;
const K1: f64 = 8e-7;
const K2: f64 = 1e-4;
const DELTA: f64 = 0.7;
const M1: f64 = 1e-5;
const M2: f64 = 1e-5;
const NT: f64 = 100.0;
const C: f64 = 13.0;
const RHO1: f64 = 1.0;
const RHO2: f64 = 1.0;
const LAMBDA_E: f64 = 1.0;
const BE: f64 = 0.3;
const KB: f64 = 100.0;
const DE: f64 = 0.25;
const KD: f64 = 500.0;
const DELTA_E: f64 = 0.1;

// Simulation parameters:
const DT: f64 = 5.0;
const SIM_STEPS: u32 = 1000;

const DT_STEP: f64 = DT / SIM_STEPS as f64;

// RL parameters:
const LIMITS: (f64, f64) = (-5.0, 8.0);
const ALL_ACTIONS: [(f64, f64); 4] = [(0.0, 0.0), (0.7, 0.0), (0.0, 0.3), (0.7, 0.3)];


#[derive(Debug, Clone, Copy)]
enum StateIndex {
    T1 = 0,
    T1S = 1,
    T2 = 2,
    T2S = 3,
    V = 4,
    E = 5,
}

impl Index<StateIndex> for Vec<f64> {
    type Output = f64;

    fn index(&self, idx: StateIndex) -> &f64 {
        self.index(idx as usize)
    }
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


pub struct HIVTreatment {
    eps: (f64, f64),
    state: Array1<f64>,
}

impl HIVTreatment {
    fn new(t1: f64, t1s: f64, t2: f64, t2s: f64, v: f64, e: f64) -> HIVTreatment {
        HIVTreatment {
            eps: ALL_ACTIONS[0],
            state: arr1(&vec![t1, t1s, t2, t2s, v, e])
        }
    }

    fn update_state(&mut self, a: usize) {
        let eps = ALL_ACTIONS[a];
        let fx = |_x, y| HIVTreatment::grad(eps, y);

        self.eps = eps;

        let mut ns = runge_kutta4(&fx, 0.0, self.state.clone(), DT_STEP);
        for _ in 1..SIM_STEPS {
            ns = runge_kutta4(&fx, 0.0, ns, DT_STEP);
        }

        self.state = ns;
    }

    fn grad(eps: (f64, f64), state: Array1<f64>) -> Array1<f64> {
        let t1 = state[StateIndex::T1];
        let t1s = state[StateIndex::T1S];
        let t2 = state[StateIndex::T2];
        let t2s = state[StateIndex::T2S];
        let v = state[StateIndex::V];
        let e = state[StateIndex::E];

        let tmp1 = (1.0 - eps.0)*K1*v*t1;
        let tmp2 = (1.0 - F*eps.0)*K2*v*t2;
        let sum_ts = t1s + t2s;

        let d_t1 = LAMBDA1 - D1*t1 - tmp1;
        let d_t1s = tmp1 - DELTA*t1s - M1*e*t1s;

        let d_t2 = LAMBDA2 - D2*t2 - tmp2;
        let d_t2s = tmp2 - DELTA*t2s - M2*e*t2s;

        let d_v = (1.0 - eps.1)*NT*DELTA*sum_ts - C*v -
            ((1.0 - eps.0)*RHO1*K1*t1 + (1.0 - F*eps.0)*RHO2*K2*t2)*v;
        let d_e = LAMBDA_E + BE*sum_ts/(sum_ts + KB)*e - DE*sum_ts/(sum_ts + KD)*e - DELTA_E*e;

        arr1(&vec![d_t1, d_t1s, d_t2, d_t2s, d_v, d_e])
    }
}

impl Default for HIVTreatment {
    fn default() -> HIVTreatment {
        HIVTreatment::new(163573.0, 11945.0, 5.0, 46.0, 63919.0, 24.0)
    }
}

impl Domain for HIVTreatment {
    type StateSpace = RegularSpace<Continuous>;
    type ActionSpace = ActionSpace;

    fn emit(&self) -> Observation<Self::StateSpace, Self::ActionSpace> {
        let s = self.state.mapv(|v| v.log10()).to_vec();

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
        false
    }

    fn reward(&self,
              _: &Observation<Self::StateSpace, Self::ActionSpace>,
              to: &Observation<Self::StateSpace, Self::ActionSpace>) -> f64
    {
        let s = to.state();
        let r = 1e3*s[StateIndex::E]
            - 0.1*s[StateIndex::V]
            - 2e4*self.eps.0.powf(2.0)
            - 2e3*self.eps.1.powf(2.0);

        r.signum()*r.abs().log10()
    }

    fn state_space(&self) -> Self::StateSpace {
        Self::StateSpace::new()
            .push(Continuous::new(LIMITS.0, LIMITS.1))
            .push(Continuous::new(LIMITS.0, LIMITS.1))
            .push(Continuous::new(LIMITS.0, LIMITS.1))
            .push(Continuous::new(LIMITS.0, LIMITS.1))
            .push(Continuous::new(LIMITS.0, LIMITS.1))
            .push(Continuous::new(LIMITS.0, LIMITS.1))
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::new(Discrete::new(3))
    }
}

#[cfg(test)]
mod tests {
}
