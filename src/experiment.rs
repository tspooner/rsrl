use agents::Agent;
use domain::{Domain, Observation, Transition};
use geometry::{Space, ActionSpace};


#[derive(Debug)]
pub struct Episode {
    pub n_steps: u64
}


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

        let mut n_steps = 0;
        for j in 1..(self.step_limit+1) {
            let t = domain.step(a);

            n_steps = j;

            a = match t.to {
                Observation::Terminal(_) => break,
                _ => self.agent.handle(&t)
            };
        }

        Some(Episode {
            n_steps: n_steps
        })
    }
}
