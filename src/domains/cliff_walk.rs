use crate::geometry::{Matrix, TwoSpace, discrete::Ordinal};
use super::{Domain, Observation, Transition, grid_world::{GridWorld, Motion}};

const ALL_ACTIONS: [Motion; 4] = [
    Motion::North(1),
    Motion::East(1),
    Motion::South(1),
    Motion::West(1),
];

pub struct CliffWalk {
    gw: GridWorld<u8>,
    loc: [usize; 2],
}

impl CliffWalk {
    pub fn new(height: usize, width: usize) -> CliffWalk {
        CliffWalk {
            gw: GridWorld::new(Matrix::zeros((height, width))),
            loc: [0; 2],
        }
    }

    fn update_state(&mut self, a: usize) {
        self.loc = self.gw.perform_motion(self.loc, ALL_ACTIONS[a]);
    }
}

impl Default for CliffWalk {
    fn default() -> CliffWalk { CliffWalk::new(5, 12) }
}

impl Domain for CliffWalk {
    type StateSpace = TwoSpace<Ordinal>;
    type ActionSpace = Ordinal;

    fn emit(&self) -> Observation<[usize; 2]> {
        if self.is_terminal() {
            Observation::Terminal(self.loc)
        } else {
            Observation::Full(self.loc)
        }
    }

    fn step(&mut self, action: usize) -> Transition<[usize; 2], usize> {
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

    fn reward(&self, _: &Observation<[usize; 2]>, to: &Observation<[usize; 2]>) -> f64 {
        match *to {
            Observation::Terminal(_) => {
                let x = to.state()[0];

                if x == self.gw.width() - 1 {
                    50.0
                } else {
                    -50.0
                }
            },
            _ => { 0.0 },
        }
    }

    fn is_terminal(&self) -> bool { self.loc[0] > 0 && self.loc[1] == 0 }

    fn state_space(&self) -> Self::StateSpace {
        TwoSpace::new([
            Ordinal::new(self.gw.width()),
            Ordinal::new(self.gw.height()),
        ])
    }

    fn action_space(&self) -> Ordinal { Ordinal::new(4) }
}

#[cfg(test)]
mod tests {
    use super::{CliffWalk, Domain};

    #[test]
    fn test_cliff_direct() {
        let mut cw = CliffWalk::default();

        cw.step(2);
        assert!(!cw.is_terminal());

        cw.step(3);
        assert!(!cw.is_terminal());

        cw.step(1);
        assert!(cw.is_terminal());
    }

    #[test]
    fn test_cliff_indirect() {
        let mut cw = CliffWalk::default();

        cw.step(0);
        cw.step(1);
        cw.step(1);
        assert!(!cw.is_terminal());

        let t = cw.step(2);

        assert!(cw.is_terminal());
        assert!(t.terminated());
        assert!(t.reward.is_sign_negative());
    }

    #[test]
    fn test_optimal() {
        let mut cw = CliffWalk::default();

        cw.step(0);
        for _ in 0..11 { cw.step(1); }
        assert!(!cw.is_terminal());

        let t = cw.step(2);

        assert!(cw.is_terminal());
        assert!(t.terminated());
        assert!(t.reward.is_sign_positive());
    }

    #[test]
    fn test_safe() {
        let mut cw = CliffWalk::default();

        for _ in 0..4 { cw.step(0); }
        for _ in 0..11 { cw.step(1); }
        assert!(!cw.is_terminal());

        cw.step(2);
        assert!(!cw.is_terminal());

        for _ in 0..2 { cw.step(2); }
        let t = cw.step(2);

        assert!(cw.is_terminal());
        assert!(t.terminated());
        assert!(t.reward.is_sign_positive());
    }
}
