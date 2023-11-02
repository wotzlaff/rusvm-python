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

fn extract_params_problem(params_dict: Option<&PyDict>) -> PyResult<smorust::problem::Params> {
    let mut params = smorust::problem::Params::new();
    if let Some(lambda) = extract::<f64>(params_dict, "lmbda")? {
        params.lambda = lambda;
    }
    if let Some(smoothing) = extract::<f64>(params_dict, "smoothing")? {
        params.smoothing = smoothing;
    }
    if let Some(max_asum) = extract::<f64>(params_dict, "max_asum")? {
        params.max_asum = max_asum;
    }
    if let Some(regularization) = extract::<f64>(params_dict, "regularization")? {
        params.regularization = regularization;
    }
    Ok(params)
}

fn extract_params_smo(params_dict: Option<&PyDict>) -> PyResult<smorust::smo::Params> {
    let mut params = smorust::smo::Params::new();
    params.tol = extract::<f64>(params_dict, "tol")?.unwrap_or(params.tol);
    params.max_steps = extract::<usize>(params_dict, "max_steps")?.unwrap_or(params.max_steps);
    params.verbose = extract::<usize>(params_dict, "verbose")?.unwrap_or(params.verbose);
    params.log_objective =
        extract::<bool>(params_dict, "log_objective")?.unwrap_or(params.log_objective);
    params.second_order =
        extract::<bool>(params_dict, "second_order")?.unwrap_or(params.second_order);
    params.shrinking_period =
        extract::<usize>(params_dict, "shrinking_period")?.unwrap_or(params.shrinking_period);
    params.shrinking_threshold =
        extract::<f64>(params_dict, "shrinking_threshold")?.unwrap_or(params.shrinking_threshold);
    params.time_limit = extract::<f64>(params_dict, "time_limit")?.unwrap_or(params.time_limit);
    Ok(params)
}

#[pymodule]
fn smorupy<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(signature = (x, y, kind = "classification", params_problem = None, params_smo = None, cache_size = 0, callback = None))]
    fn solve<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        kind: &str,
        params_problem: Option<&PyDict>,
        params_smo: Option<&PyDict>,
        cache_size: usize,
        callback: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        check_params(
            params_smo,
            vec![
                "tol",
                "max_steps",
                "verbose",
                "log_objective",
                "second_order",
                "shrinking_period",
                "shrinking_threshold",
                "time_limit",
            ]
            .as_slice(),
        )?;
        let params_smo = extract_params_smo(params_smo)?;

        let problem: Box<dyn smorust::problem::Problem> = match kind {
            "classification" => {
                check_params(
                    params_problem,
                    vec!["lmbda", "smoothing", "max_asum", "shift"].as_slice(),
                )?;
                let mut problem = smorust::problem::Classification::new(
                    y.as_slice()?,
                    extract_params_problem(params_problem)?,
                );
                if let Some(shift) = extract::<f64>(params_problem, "shift")? {
                    problem.shift = shift;
                }
                Box::new(problem)
            }
            "regression" => {
                check_params(
                    params_problem,
                    vec!["lmbda", "smoothing", "max_asum", "epsilon"].as_slice(),
                )?;
                let mut problem = smorust::problem::Regression::new(
                    y.as_slice()?,
                    extract_params_problem(params_problem)?,
                );

                if let Some(epsilon) = extract::<f64>(params_problem, "epsilon")? {
                    problem.epsilon = epsilon;
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
            &params_smo,
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
