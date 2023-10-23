use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::conversion::IntoPy;
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod kernel;
mod problem;
mod smo;
mod state;
pub use kernel::{GaussianKernel, Kernel};
pub use problem::{Classification, Problem};
use smo::{solve, SMOResult, Status};
pub use state::State;

impl IntoPy<PyObject> for SMOResult {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let res = PyDict::new(py);
        let _ = res.set_item("a", self.a);
        let _ = res.set_item("b", self.b);
        let _ = res.set_item("c", self.c);
        let _ = res.set_item("value", self.value);
        let _ = res.set_item("violation", self.violation);
        let _ = res.set_item("steps", self.steps);
        let _ = res.set_item("time", self.time);
        let _ = res.set_item(
            "status",
            match self.status {
                Status::MaxSteps => "max_steps",
                Status::Optimal => "optimal",
                Status::TimeLimit => "time_limit",
            },
        );
        res.into_py(py)
    }
}

#[pymodule]
fn smorust<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(signature = (x, y, lmbda = 1e-3, smoothing = 0.0, tol = 1e-4, max_steps = 1_000_000_000, verbose = 0, second_order = true, shrinking_period = 0, shrinking_threshold = 1.0, time_limit = 0.0))]
    fn solve_classification<'py>(
        _py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        lmbda: f64,
        smoothing: f64,
        tol: f64,
        max_steps: usize,
        verbose: usize,
        second_order: bool,
        shrinking_period: usize,
        shrinking_threshold: f64,
        time_limit: f64,
    ) -> PyResult<SMOResult> {
        let problem = Classification::new(y.to_vec()?, lmbda).set_smoothing(smoothing);
        let data = x.as_array();
        let mut kernel = GaussianKernel::new(1.0, data);
        let result = solve(
            &problem,
            &mut kernel,
            tol,
            max_steps,
            verbose,
            second_order,
            shrinking_period,
            shrinking_threshold,
            time_limit,
        );
        Ok(result)
    }
    Ok(())
}
