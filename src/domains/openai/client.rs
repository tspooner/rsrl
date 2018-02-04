use super::cpython::{Python, GILGuard, ObjectProtocol, PyModule, PyResult, PyObject, PyString};


pub struct GymClient {
    pub gil: GILGuard,
    pub gym: PyModule,
}

impl GymClient {
    pub fn new() -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let gym = gil.python().import("gym")?;

        gil.python().run("import logging; logging.getLogger('gym.envs.registration').setLevel(logging.CRITICAL)", None, None)?;

        Ok(Self {
            gil: gil,
            gym: gym,
        })
    }

    pub fn py(&self) -> Python {
        self.gil.python()
    }

    pub fn make(&mut self, env_id: &str, monitor_path: Option<String>) -> PyResult<PyObject> {
        let py = self.py();
        let maker = self.gym.get(py, "make")?;
        let mut env = maker.call(py, (PyString::new(py, env_id),), None);

        if let Some(path) = monitor_path {
            let monitor = self.gym.get(py, "wrappers")?.get(py, "Monitor")?;

            env
            
        } else {
            env
        }
    }

    pub fn monitor(&mut self, path: &str) -> PyResult<PyObject> {
        let py = self.py();

        monitor.call(py, (PyString::new(py, env_id),), None)
    }
}
