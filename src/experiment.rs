use agents::Agent;
use domain::{Domain, Observation};
use geometry::{Space, ActionSpace};


/// Container for episodic statistics.
#[derive(Debug)]
pub struct Episode {
    /// The number of steps taken to reach the terminal state.
    pub n_steps: u64,

    /// The total accumulated reward over the episode.
    pub total_reward: f64
}


/// Helper function for running experiments.
pub fn run<T>(runner: T, n_episodes: usize) -> Vec<Episode>
    where T: Iterator<Item=Episode>
{
    runner.enumerate()
          .take(n_episodes)
          .inspect(|&(i, ref res)| {
              info!("Episode {} - {} steps | reward {}",
                    i+1, res.n_steps, res.total_reward)
          })
          .map(|(_, res)| res)
          .collect::<Vec<_>>()
}


/// Utility for running a single evaluation episode.
pub struct Evaluation<'a, A: 'a, D> {
    agent: &'a mut A,
    domain_factory: Box<Fn() -> D>,
}

impl<'a, S: Space, A, D> Evaluation<'a, A, D>
    where A: Agent<S>,
          D: Domain<StateSpace=S, ActionSpace=ActionSpace>
{
    pub fn new(agent: &'a mut A,
               domain_factory: Box<Fn() -> D>) -> Evaluation<'a, A, D>
    {
        Evaluation {
            agent: agent,
            domain_factory: domain_factory,
        }
    }
}

impl<'a, S: Space, A, D> Iterator for Evaluation<'a, A, D>
    where A: Agent<S>,
          D: Domain<StateSpace=S, ActionSpace=ActionSpace>
{
    type Item = Episode;

    fn next(&mut self) -> Option<Episode> {
        let mut domain = (self.domain_factory)();
        let mut a = self.agent.pi_target(domain.emit().state());

        let mut e = Episode {
            n_steps: 1,
            total_reward: 0.0,
        };

        loop {
            let t = domain.step(a);

            e.n_steps += 1;
            e.total_reward += t.reward;

            a = match t.to {
                Observation::Terminal(_) => break,
                _ => self.agent.pi(&t.to.state())
            };
        }

        Some(e)
    }
}


/// Utility for running a sequence of training episodes.
pub struct SerialExperiment<'a, A: 'a, D> {
    agent: &'a mut A,
    domain_factory: Box<Fn() -> D>,

    step_limit: u64
}

impl<'a, S: Space, A, D> SerialExperiment<'a, A, D>
    where A: Agent<S>,
          D: Domain<StateSpace=S, ActionSpace=ActionSpace>
{
    pub fn new(agent: &'a mut A,
               domain_factory: Box<Fn() -> D>,
               step_limit: u64) -> SerialExperiment<'a, A, D>
    {
        SerialExperiment {
            agent: agent,
            domain_factory: domain_factory,
            step_limit: step_limit,
        }
    }
}

impl<'a, S: Space, A, D> Iterator for SerialExperiment<'a, A, D>
    where A: Agent<S>,
          D: Domain<StateSpace=S, ActionSpace=ActionSpace>
{
    type Item = Episode;

    fn next(&mut self) -> Option<Episode> {
        let mut domain = (self.domain_factory)();
        let mut a = self.agent.pi(domain.emit().state());

        let mut e = Episode {
            n_steps: 1,
            total_reward: 0.0,
        };

        for j in 1..(self.step_limit+1) {
            let t = domain.step(a);

            e.n_steps = j;
            e.total_reward += t.reward;

            self.agent.learn_transition(&t);

            a = match t.to {
                Observation::Terminal(_) => break,
                _ => self.agent.pi(&t.to.state())
            };
        }

        Some(e)
    }
}
