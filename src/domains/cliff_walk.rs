use geometry::{
    ActionSpace, PairSpace,
    dimensions::Discrete,
};
use super::{
    Domain, Observation, Transition,
    grid_world::{GridWorld, Motion},
};
use {Matrix};

const ALL_ACTIONS: [Motion; 4] = [
    Motion::North(1),
    Motion::East(1),
    Motion::South(1),
    Motion::West(1),
];

pub struct CliffWalk {
    gw: GridWorld<u8>,
    loc: (usize, usize),
}

impl CliffWalk {
    pub fn new(height: usize, width: usize) -> CliffWalk {
        CliffWalk {
            gw: GridWorld::new(Matrix::zeros((height, width))),
            loc: (0, 0),
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
    type StateSpace = PairSpace<Discrete, Discrete>;
    type ActionSpace = ActionSpace;

    fn emit(&self) -> Observation<(usize, usize), usize> {
        if self.is_terminal() {
            Observation::Terminal(self.loc)
        } else {
            Observation::Full {
                state: self.loc,
                actions: [0, 1, 2, 3].iter().cloned().collect(),
            }
        }
    }

    fn step(&mut self, a: usize) -> Transition<(usize, usize), usize> {
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

    fn reward(
        &self,
        from: &Observation<(usize, usize), usize>,
        to: &Observation<(usize, usize), usize>,
    ) -> f64
    {
        match to {
            &Observation::Terminal(_) => {
                if to.state().0 == self.gw.width() - 1 {
                    50.0
                } else {
                    -50.0
                }
            },
            _ => {
                if from.state() == to.state() {
                    -1.0
                } else {
                    0.0
                }
            },
        }
    }

    fn is_terminal(&self) -> bool { self.loc.0 > 0 && self.loc.1 == 0 }

    fn state_space(&self) -> Self::StateSpace {
        PairSpace::new(
            Discrete::new(self.gw.width()),
            Discrete::new(self.gw.height()),
        )
    }

    fn action_space(&self) -> ActionSpace { ActionSpace::new(Discrete::new(4)) }
}
