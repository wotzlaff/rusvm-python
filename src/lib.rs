use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::conversion::IntoPy;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn status_to_dict(status: smorust::Status, py: Python<'_>) -> PyObject {
    let dict = PyDict::new(py);
    let _ = dict.set_item("a", status.a);
    let _ = dict.set_item("b", status.b);
    let _ = dict.set_item("c", status.c);
    let _ = dict.set_item("value", status.value);
    let _ = dict.set_item("violation", status.violation);
    let _ = dict.set_item("steps", status.steps);
    let _ = dict.set_item("time", status.time);
    let _ = dict.set_item(
        "status",
        match status.code {
            smorust::StatusCode::Initialized => "initialized",
            smorust::StatusCode::MaxSteps => "max_steps",
            smorust::StatusCode::Optimal => "optimal",
            smorust::StatusCode::TimeLimit => "time_limit",
        },
    );
    dict.into_py(py)
}

#[pymodule]
fn smorupy<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(signature = (x, y, lmbda = 1e-3, smoothing = 0.0, tol = 1e-4, max_steps = 1_000_000_000, verbose = 0, second_order = true, shrinking_period = 0, shrinking_threshold = 1.0, time_limit = 0.0, cache_size = 0))]
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
        cache_size: usize,
    ) -> PyResult<PyObject> {
        let problem =
            smorust::problem::Classification::new(y.as_slice()?, lmbda).with_smoothing(smoothing);
        let data = x.as_array();
        let (mut base, mut cached);
        base = smorust::kernel::GaussianKernel::new(1.0, data);
        let kernel: &mut dyn smorust::kernel::Kernel = if cache_size > 0 {
            cached = smorust::kernel::CachedKernel::from(&mut base, cache_size);
            &mut cached
        } else {
            &mut base
        };

        let result = smorust::solve(
            &problem,
            kernel,
            tol,
            max_steps,
            verbose,
            second_order,
            shrinking_period,
            shrinking_threshold,
            time_limit,
        );
        let py_result = status_to_dict(result, py);
        Ok(py_result)
    }
    Ok(())
}
