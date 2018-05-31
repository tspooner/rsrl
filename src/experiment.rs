use agents::Controller;
use domains::{Domain, Observation};
use geometry::{dimensions::Discrete, Space};
use policies::Greedy;
use slog::{Logger, Record, Result as LogResult, Serializer, KV};

/// Container for episodic statistics.
#[derive(Debug)]
pub struct Episode {
    /// The number of steps taken to reach the terminal state.
    pub steps: u64,

    /// The total accumulated reward over the episode.
    pub reward: f64,
}

impl KV for Episode {
    fn serialize(&self, _: &Record, serializer: &mut Serializer) -> LogResult {
        serializer.emit_u64("steps", self.steps)?;
        serializer.emit_f64("reward", self.reward)?;

        Ok(())
    }
}

/// Helper function for running experiments.
pub fn run<T>(runner: T, n_episodes: usize, logger: Option<Logger>) -> Vec<Episode>
where T: Iterator<Item = Episode> {
    let exp = runner.take(n_episodes);

    match logger {
        Some(logger) => exp.zip(1..(n_episodes + 1))
            .inspect(|&(ref res, i)| {
                info!(logger, "episode {}", i; res);
            })
            .map(|(res, _)| res)
            .collect(),

        None => exp.collect(),
    }
}

/// Utility for running a single evaluation episode.
pub struct Evaluation<'a, A: 'a, D> {
    agent: &'a mut A,
    domain_factory: Box<Fn() -> D>,

    greedy: Greedy,
}

impl<'a, S: Space, A, D> Evaluation<'a, A, D>
where
    A: Controller<S::Value, usize>,
    D: Domain<StateSpace = S, ActionSpace = Discrete>,
{
    pub fn new(agent: &'a mut A, domain_factory: Box<Fn() -> D>) -> Evaluation<'a, A, D> {
        Evaluation {
            agent: agent,
            domain_factory: domain_factory,

            greedy: Greedy,
        }
    }
}

impl<'a, S: Space, A, D> Iterator for Evaluation<'a, A, D>
where
    A: Controller<S::Value, usize>,
    D: Domain<StateSpace = S, ActionSpace = Discrete>,
{
    type Item = Episode;

    fn next(&mut self) -> Option<Episode> {
        let mut domain = (self.domain_factory)();
        let mut a = self.agent
            .evaluate_policy(&mut self.greedy, &domain.emit().state());

        let mut e = Episode {
            steps: 1,
            reward: 0.0,
        };

        loop {
            let t = domain.step(a);

            e.steps += 1;
            e.reward += t.reward;

            a = match t.to {
                Observation::Terminal(_) => {
                    self.agent.handle_terminal(&t);
                    break;
                },
                _ => self.agent.evaluate_policy(&mut self.greedy, &t.to.state()),
            };
        }

        Some(e)
    }
}

/// Utility for running a sequence of training episodes.
pub struct SerialExperiment<'a, A: 'a, D> {
    agent: &'a mut A,
    domain_factory: Box<Fn() -> D>,

    step_limit: u64,
}

impl<'a, S: Space, A, D> SerialExperiment<'a, A, D>
where
    A: Controller<S::Value, usize>,
    D: Domain<StateSpace = S, ActionSpace = Discrete>,
{
    pub fn new(
        agent: &'a mut A,
        domain_factory: Box<Fn() -> D>,
        step_limit: u64,
    ) -> SerialExperiment<'a, A, D>
    {
        SerialExperiment {
            agent: agent,
            domain_factory: domain_factory,
            step_limit: step_limit,
        }
    }
}

impl<'a, S: Space, A, D> Iterator for SerialExperiment<'a, A, D>
where
    A: Controller<S::Value, usize>,
    D: Domain<StateSpace = S, ActionSpace = Discrete>,
{
    type Item = Episode;

    fn next(&mut self) -> Option<Episode> {
        let mut domain = (self.domain_factory)();
        let mut a = self.agent.mu(domain.emit().state());

        let mut e = Episode {
            steps: 1,
            reward: 0.0,
        };

        for j in 1..(self.step_limit + 1) {
            let t = domain.step(a);

            e.steps = j;
            e.reward += t.reward;

            self.agent.handle_sample(&t);

            // TODO: Clean this mess up!
            if let Observation::Terminal(_) = t.to {
                self.agent.handle_terminal(&t);
                break;
            } else if j >= self.step_limit {
                self.agent.handle_terminal(&t);
                break;
            } else {
                a = self.agent.mu(&t.to.state());
            }
        }

        Some(e)
    }
}
