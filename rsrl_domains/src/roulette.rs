use rand::{thread_rng, Rng, rngs::ThreadRng};
use crate::{Domain, Observation, Transition, spaces::{real::Reals, discrete::Ordinal}};

#[derive(Debug)]
pub struct Roulette {
    active: bool,
    reward: f64,
    wealth: f64,
    bet_size: f64,

    rng: ThreadRng,
}

impl Roulette {
    pub fn new(budget: f64, bet_size: f64) -> Self {
        Self {
            active: true,
            reward: 0.0,
            wealth: budget,
            bet_size,

            rng: thread_rng(),
        }
    }

    pub fn update_state(&mut self, action: usize) {
        if action == 156 {
            self.active = false;

            return;
        }

        let landing = self.rng.gen_range(0, 37);
        let payoff = match action {
            // Straight up:
            a @ 0 ..= 36 if a == landing => Some(35.0 * self.bet_size),

            // Splits:
            37 if landing == 0 || landing == 1 => Some(17.0 * self.bet_size),
            38 if landing == 0 || landing == 2 => Some(17.0 * self.bet_size),
            39 if landing == 0 || landing == 3 => Some(17.0 * self.bet_size),
            40 if landing == 1 || landing == 2 => Some(17.0 * self.bet_size),
            41 if landing == 2 || landing == 3 => Some(17.0 * self.bet_size),

            42 if landing == 1 || landing == 4 => Some(17.0 * self.bet_size),
            43 if landing == 2 || landing == 5 => Some(17.0 * self.bet_size),
            44 if landing == 3 || landing == 6 => Some(17.0 * self.bet_size),
            45 if landing == 4 || landing == 5 => Some(17.0 * self.bet_size),
            46 if landing == 5 || landing == 6 => Some(17.0 * self.bet_size),

            47 if landing == 4 || landing == 7 => Some(17.0 * self.bet_size),
            48 if landing == 5 || landing == 8 => Some(17.0 * self.bet_size),
            49 if landing == 6 || landing == 9 => Some(17.0 * self.bet_size),
            50 if landing == 7 || landing == 8 => Some(17.0 * self.bet_size),
            51 if landing == 8 || landing == 9 => Some(17.0 * self.bet_size),

            52 if landing == 7 || landing == 10 => Some(17.0 * self.bet_size),
            53 if landing == 8 || landing == 11 => Some(17.0 * self.bet_size),
            54 if landing == 9 || landing == 12 => Some(17.0 * self.bet_size),
            55 if landing == 10 || landing == 11 => Some(17.0 * self.bet_size),
            56 if landing == 11 || landing == 12 => Some(17.0 * self.bet_size),

            57 if landing == 10 || landing == 13 => Some(17.0 * self.bet_size),
            58 if landing == 11 || landing == 14 => Some(17.0 * self.bet_size),
            59 if landing == 12 || landing == 15 => Some(17.0 * self.bet_size),
            60 if landing == 13 || landing == 14 => Some(17.0 * self.bet_size),
            61 if landing == 14 || landing == 15 => Some(17.0 * self.bet_size),

            62 if landing == 13 || landing == 16 => Some(17.0 * self.bet_size),
            63 if landing == 14 || landing == 17 => Some(17.0 * self.bet_size),
            64 if landing == 15 || landing == 18 => Some(17.0 * self.bet_size),
            65 if landing == 16 || landing == 17 => Some(17.0 * self.bet_size),
            66 if landing == 17 || landing == 18 => Some(17.0 * self.bet_size),

            67 if landing == 16 || landing == 19 => Some(17.0 * self.bet_size),
            68 if landing == 17 || landing == 20 => Some(17.0 * self.bet_size),
            69 if landing == 18 || landing == 21 => Some(17.0 * self.bet_size),
            70 if landing == 19 || landing == 20 => Some(17.0 * self.bet_size),
            71 if landing == 20 || landing == 21 => Some(17.0 * self.bet_size),

            72 if landing == 19 || landing == 22 => Some(17.0 * self.bet_size),
            73 if landing == 20 || landing == 23 => Some(17.0 * self.bet_size),
            74 if landing == 21 || landing == 24 => Some(17.0 * self.bet_size),
            75 if landing == 22 || landing == 23 => Some(17.0 * self.bet_size),
            76 if landing == 23 || landing == 24 => Some(17.0 * self.bet_size),

            77 if landing == 22 || landing == 25 => Some(17.0 * self.bet_size),
            78 if landing == 23 || landing == 26 => Some(17.0 * self.bet_size),
            79 if landing == 24 || landing == 27 => Some(17.0 * self.bet_size),
            80 if landing == 25 || landing == 26 => Some(17.0 * self.bet_size),
            81 if landing == 26 || landing == 27 => Some(17.0 * self.bet_size),

            82 if landing == 25 || landing == 28 => Some(17.0 * self.bet_size),
            83 if landing == 26 || landing == 29 => Some(17.0 * self.bet_size),
            84 if landing == 27 || landing == 30 => Some(17.0 * self.bet_size),
            85 if landing == 28 || landing == 29 => Some(17.0 * self.bet_size),
            86 if landing == 29 || landing == 30 => Some(17.0 * self.bet_size),

            87 if landing == 28 || landing == 31 => Some(17.0 * self.bet_size),
            88 if landing == 29 || landing == 32 => Some(17.0 * self.bet_size),
            89 if landing == 30 || landing == 33 => Some(17.0 * self.bet_size),
            90 if landing == 31 || landing == 32 => Some(17.0 * self.bet_size),
            91 if landing == 32 || landing == 33 => Some(17.0 * self.bet_size),

            92 if landing == 31 || landing == 34 => Some(17.0 * self.bet_size),
            93 if landing == 32 || landing == 35 => Some(17.0 * self.bet_size),
            94 if landing == 33 || landing == 36 => Some(17.0 * self.bet_size),
            95 if landing == 34 || landing == 35 => Some(17.0 * self.bet_size),
            96 if landing == 35 || landing == 36 => Some(17.0 * self.bet_size),

            // Streets:
            97 if landing == 0 || landing == 1 || landing == 2 => Some(11.0 * self.bet_size),
            98 if landing == 0 || landing == 2 || landing == 3 => Some(11.0 * self.bet_size),
            99 if landing >= 1 && landing <= 3  => Some(11.0 * self.bet_size),
            100 if landing >= 4 && landing <= 6 => Some(11.0 * self.bet_size),
            101 if landing >= 7 && landing <= 9 => Some(11.0 * self.bet_size),
            102 if landing >= 10 && landing <= 12 => Some(11.0 * self.bet_size),
            103 if landing >= 13 && landing <= 15 => Some(11.0 * self.bet_size),
            104 if landing >= 16 && landing <= 18 => Some(11.0 * self.bet_size),
            105 if landing >= 19 && landing <= 21 => Some(11.0 * self.bet_size),
            106 if landing >= 22 && landing <= 24 => Some(11.0 * self.bet_size),
            107 if landing >= 25 && landing <= 27 => Some(11.0 * self.bet_size),
            108 if landing >= 28 && landing <= 30 => Some(11.0 * self.bet_size),
            109 if landing >= 31 && landing <= 33 => Some(11.0 * self.bet_size),
            110 if landing >= 34 && landing <= 36 => Some(11.0 * self.bet_size),

            // Top line:
            111 if landing == 0 || landing == 1 || landing == 2 || landing == 3 => Some(8.0 * self.bet_size),

            // Corners:
            112 if landing == 1 || landing == 2 || landing == 4 || landing == 5 => Some(8.0 * self.bet_size),

            113 if landing == 2 || landing == 3 || landing == 5 || landing == 6 => Some(8.0 * self.bet_size),
            114 if landing == 5 || landing == 6 || landing == 8 || landing == 9 => Some(8.0 * self.bet_size),

            115 if landing == 7 || landing == 8 || landing == 10 || landing == 11 => Some(8.0 * self.bet_size),
            116 if landing == 8 || landing == 9 || landing == 11 || landing == 12 => Some(8.0 * self.bet_size),

            117 if landing == 10 || landing == 11 || landing == 13 || landing == 14 => Some(8.0 * self.bet_size),
            118 if landing == 11 || landing == 12 || landing == 14 || landing == 15 => Some(8.0 * self.bet_size),

            119 if landing == 13 || landing == 14 || landing == 16 || landing == 17 => Some(8.0 * self.bet_size),
            120 if landing == 14 || landing == 15 || landing == 17 || landing == 18 => Some(8.0 * self.bet_size),

            121 if landing == 16 || landing == 17 || landing == 19 || landing == 20 => Some(8.0 * self.bet_size),
            122 if landing == 17 || landing == 18 || landing == 20 || landing == 21 => Some(8.0 * self.bet_size),

            123 if landing == 19 || landing == 20 || landing == 22 || landing == 23 => Some(8.0 * self.bet_size),
            124 if landing == 20 || landing == 21 || landing == 23 || landing == 24 => Some(8.0 * self.bet_size),

            125 if landing == 22 || landing == 23 || landing == 25 || landing == 26 => Some(8.0 * self.bet_size),
            126 if landing == 23 || landing == 24 || landing == 27 || landing == 27 => Some(8.0 * self.bet_size),

            127 if landing == 25 || landing == 26 || landing == 28 || landing == 29 => Some(8.0 * self.bet_size),
            128 if landing == 26 || landing == 27 || landing == 29 || landing == 30 => Some(8.0 * self.bet_size),

            129 if landing == 28 || landing == 29 || landing == 31 || landing == 32 => Some(8.0 * self.bet_size),
            130 if landing == 29 || landing == 30 || landing == 32 || landing == 33 => Some(8.0 * self.bet_size),

            131 if landing == 31 || landing == 32 || landing == 34 || landing == 35 => Some(8.0 * self.bet_size),
            132 if landing == 32 || landing == 33 || landing == 35 || landing == 36 => Some(8.0 * self.bet_size),

            // Lines:
            133 if landing >= 1 && landing <= 6 => Some(5.0 * self.bet_size),
            134 if landing >= 4 && landing <= 9 => Some(5.0 * self.bet_size),
            135 if landing >= 7 && landing <= 12 => Some(5.0 * self.bet_size),
            136 if landing >= 10 && landing <= 15 => Some(5.0 * self.bet_size),
            137 if landing >= 13 && landing <= 18 => Some(5.0 * self.bet_size),
            138 if landing >= 16 && landing <= 21 => Some(5.0 * self.bet_size),
            139 if landing >= 19 && landing <= 24 => Some(5.0 * self.bet_size),
            140 if landing >= 22 && landing <= 27 => Some(5.0 * self.bet_size),
            141 if landing >= 25 && landing <= 30 => Some(5.0 * self.bet_size),
            142 if landing >= 28 && landing <= 33 => Some(5.0 * self.bet_size),
            143 if landing >= 31 && landing <= 36 => Some(5.0 * self.bet_size),

            // Columns:
            144 if landing > 0 && landing % 3 == 1 => Some(2.0 * self.bet_size),
            145 if landing > 0 && landing % 3 == 2 => Some(2.0 * self.bet_size),
            146 if landing > 0 && landing % 3 == 0 => Some(2.0 * self.bet_size),

            // Dozens:
            147 if landing >= 1 && landing <= 12 => Some(2.0 * self.bet_size),
            148 if landing >= 13 && landing <= 24 => Some(2.0 * self.bet_size),
            149 if landing >= 25 && landing <= 36 => Some(2.0 * self.bet_size),

            // Colours:
            150 | 151 => {
                let is_red = landing == 1 || landing == 3 || landing == 5 || landing == 7 ||
                    landing == 9 || landing == 12 || landing == 14 || landing == 16 ||
                    landing == 18 || landing == 19 || landing == 21 || landing == 23 ||
                    landing == 25 || landing == 27 || landing == 30 || landing == 32 ||
                    landing == 35 || landing == 36;

                if (action == 148 && is_red) || (action == 149 && !is_red) {
                    Some(2.0 * self.bet_size)
                } else {
                    None
                }
            },

            // Odds/evens:
            152 if landing > 0 && landing % 2 == 0 => Some(self.bet_size),
            153 if landing > 0 && landing % 2 == 1 => Some(self.bet_size),

            // Halves:
            154 if landing >= 1 && landing <= 18 => Some(self.bet_size),
            155 if landing >= 19 && landing <= 36 => Some(self.bet_size),

            _ => None,
        };

        if let Some(p) = payoff {
            self.reward = p + self.bet_size;
            self.wealth += p + self.bet_size;
        } else {
            self.reward = -self.bet_size;
            self.wealth -= self.bet_size;
        }

        if self.wealth <= 1e-5 {
            self.active = false;
        }
    }
}

impl Default for Roulette {
    fn default() -> Roulette { Roulette::new(1.0, 1.0) }
}

impl Domain for Roulette {
    type StateSpace = Reals;
    type ActionSpace = Ordinal;

    fn emit(&self) -> Observation<f64> {
        if self.active {
            Observation::Full(self.wealth)
        } else {
            Observation::Terminal(self.wealth)
        }
    }

    fn step(&mut self, action: usize) -> Transition<f64, usize> {
        let from = self.emit();

        self.update_state(action);

        Transition {
            from,
            action,
            reward: self.reward,
            to: self.emit(),
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        Reals
    }

    fn action_space(&self) -> Self::ActionSpace {
        Ordinal::new(157)
    }
}
