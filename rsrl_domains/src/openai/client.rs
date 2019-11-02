use super::cpython::{GILGuard, ObjectProtocol, Python};
use super::cpython::{PyDict, PyModule, PyObject, PyResult, PyString};

pub struct GymClient {
    pub gil: GILGuard,
    pub gym: PyModule,
}

impl GymClient {
    pub fn new() -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let gym = gil.python().import("gym")?;

        gil.python().run(
            "import logging; logging.getLogger('gym.envs.registration').setLevel(logging.CRITICAL)",
            None,
            None,
        )?;

        Ok(Self { gil: gil, gym: gym })
    }

    pub fn py(&self) -> Python { self.gil.python() }

    pub fn make(&self, env_id: &str) -> PyResult<PyObject> {
        let py = self.py();

        self.gym
            .get(py, "make")?
            .call(py, (PyString::new(py, env_id),), None)
    }

    pub fn monitor(&self, env: PyObject, monitor_path: &str) -> PyResult<PyObject> {
        let py = self.py();
        let args = (env, PyString::new(py, monitor_path));

        py.import("gym.wrappers")?
            .get(py, "Monitor")?
            .call(py, args, None)
    }

    pub fn upload(&self, file_path: &str, api_key: &str) -> PyResult<PyObject> {
        let py = self.py();

        let kwargs = PyDict::new(py);
        kwargs.set_item(py, "api_key", api_key)?;

        self.gym.call(py, "upload", (file_path,), Some(&kwargs))
    }
}
