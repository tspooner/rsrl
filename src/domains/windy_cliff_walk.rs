use crate::core::Matrix;
use crate::geometry::{TwoSpace, discrete::Ordinal};
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Binomial};
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

pub struct WindyCliffWalk {
    gw: GridWorld<u8>,
    loc: [usize; 2],
    max_prob: f64,
}

impl WindyCliffWalk {
    pub fn new(height: usize, width: usize, max_prob: f64) -> WindyCliffWalk {
        WindyCliffWalk {
            gw: GridWorld::new(Matrix::zeros((height, width))),
            loc: [0; 2],
            max_prob,
        }
    }

    fn update_state(&mut self, a: usize) {
        self.loc = self.gw.perform_motion(self.loc, ALL_ACTIONS[a]);

        let n_wind = thread_rng().sample(Binomial::new(2, self.max_prob).unwrap());
        self.loc = self.gw.move_south(self.loc, n_wind as usize);

        // let width = self.gw.width();

        // if self.loc[0] > 0 && self.loc[0] < width - 1 {
            // let wind_prob = self.max_prob * self.loc[0] as f64 / (width - 2) as f64;

            // self.loc = self.gw.perform_motion(self.loc, Motion::South(wind_prob.trunc() as usize));
            // if thread_rng().gen_bool(wind_prob.fract()) {
                // self.loc = self.gw.perform_motion(self.loc, Motion::South(1));
            // }
        // }

        // if self.loc[0] > 0 && self.loc[0] < width - 1 && self.loc[1] == 0 {
            // self.loc = [0; 2];
        // }
    }
}

impl Default for WindyCliffWalk {
    fn default() -> WindyCliffWalk { WindyCliffWalk::new(5, 12, 1.0) }
}

impl Domain for WindyCliffWalk {
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

                if x < self.gw.width() - 1 {
                    -50.0
                } else {
                    50.0
                }
            },
            _ => { 0.0 },
        }
        // match *to {
            // Observation::Terminal(_) => { 100.0 },
            // _ => {
                // let f = from.state();
                // let t = to.state();

                // if f != t && t == &[0, 0] {
                    // -10.0
                // } else {
                    // -1.0
                // }
            // },
        // }
    }

    fn is_terminal(&self) -> bool {
        // self.loc[0] == self.gw.width() - 1 && self.loc[1] == 0
        self.loc[0] > 0 && self.loc[1] == 0
    }

    fn state_space(&self) -> Self::StateSpace {
        TwoSpace::new([
            Ordinal::new(self.gw.width()),
            Ordinal::new(self.gw.height()),
        ])
    }

    fn action_space(&self) -> Ordinal { Ordinal::new(8) }
}
