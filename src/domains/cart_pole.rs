use crate::{
    consts::{FOUR_THIRDS, G, TWELVE_DEGREES},
    spaces::{
        ProductSpace,
        real::Interval,
        discrete::Ordinal,
    },
};
use super::{runge_kutta4, Domain, Observation, Transition};

const DT: f64 = 0.02;

const CART_MASS: f64 = 1.0;
const CART_FORCE: f64 = 10.0;

const POLE_COM: f64 = 0.5;
const POLE_MASS: f64 = 0.1;
const POLE_MOMENT: f64 = POLE_COM * POLE_MASS;

const TOTAL_MASS: f64 = CART_MASS + POLE_MASS;

const LIMITS_X: [f64; 2] = [-2.4, 2.4];
const LIMITS_DX: [f64; 2] = [-6.0, 6.0];
const LIMITS_THETA: [f64; 2] = [-TWELVE_DEGREES, TWELVE_DEGREES];
const LIMITS_DTHETA: [f64; 2] = [-2.0, 2.0];

const REWARD_STEP: f64 = 0.0;
const REWARD_TERMINAL: f64 = -1.0;

const ALL_ACTIONS: [f64; 2] = [-1.0 * CART_FORCE, 1.0 * CART_FORCE];

make_index!(StateIndex [
    X => 0, DX => 1, THETA => 2, DTHETA => 3
]);


pub struct CartPole([f64; 4]);

impl CartPole {
    fn new(x: f64, dx: f64, theta: f64, dtheta: f64) -> CartPole {
        CartPole([x, dx, theta, dtheta])
    }

    fn update_state(&mut self, a: usize) {
        let fx = |_x, y| CartPole::grad(ALL_ACTIONS[a], y);

        let ns = runge_kutta4(&fx, 0.0, self.0.to_vec(), DT);

        self.0[StateIndex::X] = clip!(LIMITS_X[0], ns[StateIndex::X], LIMITS_X[1]);
        self.0[StateIndex::DX] = clip!(LIMITS_DX[0], ns[StateIndex::DX], LIMITS_DX[1]);

        self.0[StateIndex::THETA] =
            clip!(LIMITS_THETA[0], ns[StateIndex::THETA], LIMITS_THETA[1]);
        self.0[StateIndex::DTHETA] =
            clip!(LIMITS_DTHETA[0], ns[StateIndex::DTHETA], LIMITS_DTHETA[1]);
    }

    fn grad(force: f64, mut buffer: Vec<f64>) -> Vec<f64> {
        let dx = buffer[StateIndex::DX];
        let theta = buffer[StateIndex::THETA];
        let dtheta = buffer[StateIndex::DTHETA];

        buffer[StateIndex::X] = dx;
        buffer[StateIndex::THETA] = dtheta;

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let z = (force + POLE_MOMENT * dtheta * dtheta * sin_theta) / TOTAL_MASS;

        let numer = G * sin_theta - cos_theta * z;
        let denom = FOUR_THIRDS * POLE_COM - POLE_MOMENT * cos_theta * cos_theta;

        buffer[StateIndex::DTHETA] = numer / denom;
        buffer[StateIndex::DX] = z - POLE_COM * buffer[StateIndex::DTHETA] * cos_theta;

        buffer
    }
}

impl Default for CartPole {
    fn default() -> CartPole { CartPole::new(0.0, 0.0, 0.0, 0.0) }
}

impl Domain for CartPole {
    type StateSpace = ProductSpace<Interval>;
    type ActionSpace = Ordinal;

    fn emit(&self) -> Observation<Vec<f64>> {
        let x = self.0[StateIndex::X];
        let theta = self.0[StateIndex::THETA];

        let is_terminal =
            x <= LIMITS_X[0] ||
            x >= LIMITS_X[1] ||
            theta <= LIMITS_THETA[0] ||
            theta >= LIMITS_THETA[1];

        if is_terminal {
            Observation::Terminal(self.0.to_vec())
        } else {
            Observation::Full(self.0.to_vec())
        }
    }

    fn step(&mut self, action: usize) -> Transition<Vec<f64>, usize> {
        let from = self.emit();

        self.update_state(action);

        let to = self.emit();

        Transition {
            from,
            action,
            reward: if to.is_terminal() { REWARD_TERMINAL } else { REWARD_STEP },
            to,
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        ProductSpace::empty()
            + Interval::bounded(LIMITS_X[0], LIMITS_X[1])
            + Interval::bounded(LIMITS_DX[0], LIMITS_DX[1])
            + Interval::bounded(LIMITS_THETA[0], LIMITS_THETA[1])
            + Interval::bounded(LIMITS_DTHETA[0], LIMITS_DTHETA[1])
    }

    fn action_space(&self) -> Ordinal { Ordinal::new(2) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::{Domain, Observation};

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
