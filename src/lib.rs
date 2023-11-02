use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::conversion::IntoPy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn status_to_dict(status: &smorust::Status, py: Python<'_>) -> PyObject {
    let dict = PyDict::new(py);
    let _ = dict.set_item("a", &status.a);
    let _ = dict.set_item("b", status.b);
    let _ = dict.set_item("c", status.c);
    let _ = dict.set_item("ka", &status.ka);
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
            smorust::StatusCode::Callback => "callback",
        },
    );
    dict.into_py(py)
}

fn check_params(params: Option<&PyDict>, possible_keys: &[&str]) -> PyResult<()> {
    match params {
        Some(p) => {
            for key in p.keys() {
                let key_str = key.str()?.to_str()?;
                if !possible_keys.contains(&key_str) {
                    return Err(PyValueError::new_err(format!(
                        "unkown key '{key_str}' in params"
                    )));
                }
            }
        }
        None => {}
    };
    Ok(())
}

fn extract<'a, T: pyo3::FromPyObject<'a>>(
    params: Option<&'a PyDict>,
    key: &str,
) -> PyResult<Option<T>> {
    match params {
        Some(p) => {
            let lmbda = p.get_item(key)?;
            if lmbda.is_none() {
                Ok(None)
            } else {
                let val = lmbda.unwrap().extract::<T>()?;
                Ok(Some(val))
            }
        }
        None => Ok(None),
    }
}

fn extract_params(params: Option<&PyDict>) -> PyResult<smorust::problem::Params> {
    let mut params_problem = smorust::problem::Params::new();
    if let Some(lambda) = extract::<f64>(params, "lmbda")? {
        params_problem.lambda = lambda;
    }
    if let Some(smoothing) = extract::<f64>(params, "smoothing")? {
        params_problem.smoothing = smoothing;
    }
    if let Some(max_asum) = extract::<f64>(params, "max_asum")? {
        params_problem.max_asum = max_asum;
    }
    if let Some(regularization) = extract::<f64>(params, "regularization")? {
        params_problem.regularization = regularization;
    }
    Ok(params_problem)
}

#[pymodule]
fn smorupy<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(signature = (x, y, kind = "classification", params = None, tol = 1e-4, max_steps = 1_000_000_000, verbose = 0, log_objective = false, second_order = true, shrinking_period = 0, shrinking_threshold = 1.0, time_limit = 0.0, cache_size = 0, callback = None))]
    fn solve_classification<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        kind: &str,
        params: Option<&PyDict>,
        tol: f64,
        max_steps: usize,
        verbose: usize,
        log_objective: bool,
        second_order: bool,
        shrinking_period: usize,
        shrinking_threshold: f64,
        time_limit: f64,
        cache_size: usize,
        callback: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let problem: Box<dyn smorust::problem::Problem> = match kind {
            "classification" => {
                check_params(
                    params,
                    vec!["lmbda", "smoothing", "max_asum", "shift"].as_slice(),
                )?;
                let mut problem =
                    smorust::problem::Classification::new(y.as_slice()?, extract_params(params)?);

                if let Some(p) = params {
                    let shift = p.get_item("shift")?;
                    if !shift.is_none() {
                        let shift_value = shift.unwrap().extract::<f64>()?;
                        problem.shift = shift_value;
                    }
                }
                Box::new(problem)
            }
            "regression" => {
                check_params(
                    params,
                    vec!["lmbda", "smoothing", "max_asum", "epsilon"].as_slice(),
                )?;
                let mut problem =
                    smorust::problem::Regression::new(y.as_slice()?, extract_params(params)?);

                if let Some(p) = params {
                    let epsilon = p.get_item("epsilon")?;
                    if !epsilon.is_none() {
                        let epsilon_value = epsilon.unwrap().extract::<f64>()?;
                        problem.epsilon = epsilon_value;
                    }
                }
                Box::new(problem)
            }
            &_ => {
                return Err(PyValueError::new_err(format!(
                    "unkown problem kind '{kind}'"
                )));
            }
        };
        let data = x.as_array();
        let mut base = Box::new(smorust::kernel::GaussianKernel::new(1.0, data));
        let mut kernel: Box<dyn smorust::kernel::Kernel> = {
            if cache_size > 0 {
                Box::new(smorust::kernel::CachedKernel::from(
                    base.as_mut(),
                    cache_size,
                ))
            } else {
                base
            }
        };

        let fun: Box<dyn Fn(&smorust::Status) -> bool>;

        let result = smorust::solve(
            problem.as_ref(),
            kernel.as_mut(),
            tol,
            max_steps,
            verbose,
            log_objective,
            second_order,
            shrinking_period,
            shrinking_threshold,
            time_limit,
            match callback {
                None => {
                    fun = Box::new(|_| match py.check_signals() {
                        Err(_) => true,
                        Ok(()) => false,
                    });
                    Some(fun.as_ref())
                }
                Some(cb) => {
                    if !cb.is_callable() {
                        return Err(PyValueError::new_err("callback is not callable"));
                    } else {
                        fun = Box::new(|status: &smorust::Status| {
                            if let Err(_) = py.check_signals() {
                                return true;
                            }
                            let status_dict: PyObject = status_to_dict(status, py);
                            let ret = cb.call((status_dict,), None).unwrap();
                            let val = ret.extract::<bool>();
                            match val {
                                Ok(b) => b,
                                Err(_) => false,
                            }
                        });
                        Some(fun.as_ref())
                    }
                }
            },
        );
        let py_result = status_to_dict(&result, py);
        Ok(py_result)
    }
    Ok(())
}
