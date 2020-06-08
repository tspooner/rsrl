use super::{
    grid_world::{GridWorld, Motion},
    Domain,
    Observation,
    Reward,
};
use crate::spaces::{discrete::Ordinal, TwoSpace};
use ndarray::Array2;

const ALL_ACTIONS: [Motion; 4] = [
    Motion::North(1),
    Motion::East(1),
    Motion::South(1),
    Motion::West(1),
];

pub struct CliffWalk {
    gw: GridWorld<()>,
    loc: [usize; 2],
}

impl CliffWalk {
    pub fn new(height: usize, width: usize) -> CliffWalk {
        let gw = Array2::from_elem((height, width), ());

        CliffWalk {
            gw: GridWorld::new(gw),
            loc: [0; 2],
        }
    }
}

impl Default for CliffWalk {
    fn default() -> CliffWalk { CliffWalk::new(5, 12) }
}

impl Domain for CliffWalk {
    type StateSpace = TwoSpace<Ordinal>;
    type ActionSpace = Ordinal;

    fn emit(&self) -> Observation<[usize; 2]> {
        if self.loc[0] > 0 && self.loc[1] == 0 {
            Observation::Terminal(self.loc)
        } else {
            Observation::Full(self.loc)
        }
    }

    fn step(&mut self, action: &usize) -> (Observation<[usize; 2]>, Reward) {
        self.loc = self.gw.perform_motion(self.loc, ALL_ACTIONS[*action]);

        let to = self.emit();

        (
            to,
            match to {
                Observation::Terminal(s) if s[0] == self.gw.width() - 1 => 50.0,
                Observation::Terminal(_) => -50.0,
                _ => 0.0,
            },
        )
    }

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

        cw.step(&2);
        assert!(!cw.emit().is_terminal());

        cw.step(&3);
        assert!(!cw.emit().is_terminal());

        cw.step(&1);
        assert!(cw.emit().is_terminal());
    }

    #[test]
    fn test_cliff_indirect() {
        let mut cw = CliffWalk::default();

        cw.step(&0);
        cw.step(&1);
        cw.step(&1);
        assert!(!cw.emit().is_terminal());

        let (ns, r) = cw.step(&2);

        assert!(ns.is_terminal());
        assert!(r.is_sign_negative());
    }

    #[test]
    fn test_optimal() {
        let mut cw = CliffWalk::default();

        cw.step(&0);
        for _ in 0..11 {
            cw.step(&1);
        }
        assert!(!cw.emit().is_terminal());

        let (ns, r) = cw.step(&2);

        assert!(ns.is_terminal());
        assert!(r.is_sign_positive());
    }

    #[test]
    fn test_safe() {
        let mut cw = CliffWalk::default();

        for _ in 0..4 {
            cw.step(&0);
        }
        for _ in 0..11 {
            cw.step(&1);
        }
        assert!(!cw.emit().is_terminal());

        cw.step(&2);
        assert!(!cw.emit().is_terminal());

        for _ in 0..2 {
            cw.step(&2);
        }
        let (ns, r) = cw.step(&2);

        assert!(ns.is_terminal());
        assert!(r.is_sign_positive());
    }
}
