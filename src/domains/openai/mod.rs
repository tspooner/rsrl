extern crate cpython;

use geometry::{
    dimensions::{Continuous, Discrete},
    RegularSpace,
};
use self::cpython::{NoArgs, ObjectProtocol, PyObject, PyResult, Python};
use std::{collections::HashSet, f64};
use super::{Domain, Observation, Transition};

mod client;
use self::client::GymClient;

pub struct OpenAIGym {
    client: GymClient,
    monitor_path: Option<String>,

    env: PyObject,
    state: Vec<f64>,
    terminal: bool,
    last_reward: f64,
}

impl OpenAIGym {
    pub fn new(env_id: &str, monitor_path: Option<String>) -> PyResult<Self> {
        let client = GymClient::new()?;

        let env = if let Some(ref path) = monitor_path {
            let env = client.make(env_id)?;

            client.monitor(env, path)?
        } else {
            client.make(env_id)?
        };

        let obs = env.call_method(client.py(), "reset", NoArgs, None)?;
        let state = OpenAIGym::parse_vec(client.py(), &obs);

        Ok(Self {
            client: client,
            monitor_path: monitor_path,

            env: env,
            state: state,
            terminal: false,
            last_reward: 0.0,
        })
    }

    pub fn upload<T: Into<String>>(&self, api_key: T) -> Result<(), &'static str> {
        if let Some(ref path) = self.monitor_path {
            match self.env
                .call_method(self.client.py(), "close", NoArgs, None)
                .and_then(|_| self.client.upload(path, &api_key.into()))
            {
                Ok(_) => Ok(()),
                Err(_) => Err("upload failed - python error"),
            }
        } else {
            Err("upload failed - no monitor file")
        }
    }

    fn parse_vec(py: Python, vals: &PyObject) -> Vec<f64> {
        (0..vals.len(py).unwrap())
            .map(|i| vals.get_item(py, i).unwrap().extract::<f64>(py).unwrap())
            .collect()
    }

    fn update_state(&mut self, a: usize) {
        let py = self.client.py();

        let tr = self.env.call_method(py, "step", (a,), None).unwrap();
        let obs = tr.get_item(py, 0).unwrap();

        self.state = OpenAIGym::parse_vec(py, &obs);
        self.terminal = tr.get_item(py, 2).unwrap().extract::<bool>(py).unwrap();
        self.last_reward = tr.get_item(py, 1).unwrap().extract::<f64>(py).unwrap();
    }
}

impl Domain for OpenAIGym {
    type StateSpace = RegularSpace<Continuous>;
    type ActionSpace = Discrete;

    fn emit(&self) -> Observation<Vec<f64>, usize> {
        if self.is_terminal() {
            Observation::Terminal(self.state.clone())
        } else {
            Observation::Full {
                state: self.state.clone(),
                actions: HashSet::new(),
            }
        }
    }

    fn step(&mut self, a: usize) -> Transition<Vec<f64>, usize> {
        let from = self.emit();

        self.update_state(a);
        let to = self.emit();

        Transition {
            from: from,
            action: a,
            reward: self.last_reward,
            to: to,
        }
    }

    fn is_terminal(&self) -> bool { self.terminal }

    fn reward(&self, _: &Observation<Vec<f64>, usize>, _: &Observation<Vec<f64>, usize>) -> f64 {
        self.last_reward
    }

    fn state_space(&self) -> Self::StateSpace {
        let py = self.client.py();
        let ss = self.env.getattr(py, "observation_space").unwrap();

        let lbs = ss.getattr(py, "low").unwrap();
        let ubs = ss.getattr(py, "high").unwrap();
        let len = ss.getattr(py, "shape")
            .unwrap()
            .get_item(py, 0)
            .unwrap()
            .extract::<usize>(py)
            .unwrap();

        (0..len).fold(RegularSpace::empty(), |acc, i| {
            let lb = lbs.get_item(py, i).unwrap().extract::<f64>(py).unwrap();
            let ub = ubs.get_item(py, i).unwrap().extract::<f64>(py).unwrap();

            if lb.abs() <= 340282346638528860000000000000000000000.0
                || ub.abs() >= 340282346638528860000000000000000000000.0
            {
                acc.push(Continuous::new(f64::NEG_INFINITY, f64::INFINITY))
            } else {
                acc.push(Continuous::new(lb, ub))
            }
        })
    }

    fn action_space(&self) -> Discrete {
        let py = self.client.py();
        let n = self.env
            .getattr(py, "action_space")
            .unwrap()
            .getattr(py, "n")
            .unwrap()
            .extract::<usize>(py)
            .unwrap();

        Discrete::new(n)
    }
}
