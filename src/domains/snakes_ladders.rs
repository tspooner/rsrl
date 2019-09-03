use crate::core::Matrix;
use crate::geometry::{TwoSpace, discrete::Ordinal};
use rand::{thread_rng, Rng};
use rand_distr::{Normal, Gamma};
use super::{
    grid_world::{GridWorld, Motion},
    Domain,
    Observation,
    Transition,
};

const ALL_ACTIONS: [Motion; 8] = [
    Motion::North(1),
    Motion::East(1),
    Motion::South(1),
    Motion::West(1),
    Motion::NorthEast(1),
    Motion::NorthWest(1),
    Motion::SouthEast(1),
    Motion::SouthWest(1),
];

pub struct SnakesAndLadders {
    gw: GridWorld<u8>,
    pub loc: [usize; 2],
    reward: f64,
}

impl SnakesAndLadders {
    pub fn new(height: usize, width: usize, start_loc: [usize; 2]) -> SnakesAndLadders {
        SnakesAndLadders {
            gw: GridWorld::new(Matrix::zeros((height, width))),
            loc: start_loc,
            reward: 0.0,
        }
    }

    fn update_state(&mut self, a: usize) {
        self.reward = 0.0;
        self.loc = self.gw.perform_motion(self.loc, ALL_ACTIONS[a]);

        let w = self.gw.width();

        if self.loc[0] == w - 1 {
            self.reward = thread_rng().sample(Normal::new(1.0, 2.0f64.sqrt()).unwrap());
        } else if self.loc[0] > 0 {
            // Top row:
            if self.loc[1] == self.gw.height() - 1 {
                let mut rng = thread_rng();

                if rng.gen_bool((1.0 * self.loc[0] as f64).min(1.0)) {
                    self.reward = thread_rng().sample(Gamma::new(6.0, 0.5).unwrap());
                    self.loc[0] = w - 1;
                } else {
                    self.reward = -1.0;
                    self.loc[0] = 0;
                }
            }
            // Bottom row:
            else if self.loc[1] == 0 {
                let mut rng = thread_rng();

                if rng.gen_bool((1.0 * self.loc[0] as f64).min(1.0)) {
                    self.reward = thread_rng().sample(Normal::new(4.0, 6.0f64.sqrt()).unwrap());
                    self.loc[0] = w - 1;
                } else {
                    self.reward = -1.0;
                    self.loc[0] = 0;
                }
            }
        }
    }
}

impl Default for SnakesAndLadders {
    fn default() -> SnakesAndLadders { SnakesAndLadders::new(5, 12, [0, 2]) }
}

impl Domain for SnakesAndLadders {
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
        self.reward
    }

    fn is_terminal(&self) -> bool { self.loc[0] == self.gw.width() - 1 }

    fn state_space(&self) -> Self::StateSpace {
        TwoSpace::new([
            Ordinal::new(self.gw.width()),
            Ordinal::new(self.gw.height()),
        ])
    }

    fn action_space(&self) -> Ordinal { Ordinal::new(8) }
}
