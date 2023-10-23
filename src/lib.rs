use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::conversion::IntoPy;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use smorust::{solve, Classification, GaussianKernel, SMOResult, Status};

fn result_to_dict(res: SMOResult, py: Python<'_>) -> PyObject {
    let dict = PyDict::new(py);
    let _ = dict.set_item("a", res.a);
    let _ = dict.set_item("b", res.b);
    let _ = dict.set_item("c", res.c);
    let _ = dict.set_item("value", res.value);
    let _ = dict.set_item("violation", res.violation);
    let _ = dict.set_item("steps", res.steps);
    let _ = dict.set_item("time", res.time);
    let _ = dict.set_item(
        "status",
        match res.status {
            Status::MaxSteps => "max_steps",
            Status::Optimal => "optimal",
            Status::TimeLimit => "time_limit",
        },
    );
    dict.into_py(py)
}

#[pymodule]
fn smorupy<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(signature = (x, y, lmbda = 1e-3, smoothing = 0.0, tol = 1e-4, max_steps = 1_000_000_000, verbose = 0, second_order = true, shrinking_period = 0, shrinking_threshold = 1.0, time_limit = 0.0))]
    fn solve_classification<'py>(
        py: Python<'py>,
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
    ) -> PyResult<PyObject> {
        let problem = Classification::new(y.as_slice()?, lmbda).set_smoothing(smoothing);
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
        let py_result = result_to_dict(result, py);
        Ok(py_result)
    }
    Ok(())
}
