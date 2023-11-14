use pyo3::conversion::IntoPy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub fn status_to_dict(status: &rusvm::Status, py: Python<'_>) -> PyObject {
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
            rusvm::StatusCode::Initialized => "initialized",
            rusvm::StatusCode::MaxSteps => "max_steps",
            rusvm::StatusCode::Optimal => "optimal",
            rusvm::StatusCode::TimeLimit => "time_limit",
            rusvm::StatusCode::Callback => "callback",
            rusvm::StatusCode::NoStepPossible => "no_step_possible",
        },
    );
    dict.into_py(py)
}

pub fn check_params(params: Option<&PyDict>, possible_keys: &[&str]) -> PyResult<()> {
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

pub fn extract<'a, T: pyo3::FromPyObject<'a>>(
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

pub fn extract_params_problem(params_dict: Option<&PyDict>) -> PyResult<rusvm::problem::Params> {
    let mut params = rusvm::problem::Params::new();
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

pub fn extract_params_smo(params_dict: Option<&PyDict>) -> PyResult<rusvm::smo::Params> {
    check_params(
        params_dict,
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

    let mut params = rusvm::smo::Params::new();
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

pub fn extract_params_newton(params_dict: Option<&PyDict>) -> PyResult<rusvm::newton::Params> {
    check_params(
        params_dict,
        vec!["tol", "max_steps", "verbose", "time_limit"].as_slice(),
    )?;

    let mut params = rusvm::newton::Params::new();
    params.tol = extract::<f64>(params_dict, "tol")?.unwrap_or(params.tol);
    params.max_steps = extract::<usize>(params_dict, "max_steps")?.unwrap_or(params.max_steps);
    params.verbose = extract::<usize>(params_dict, "verbose")?.unwrap_or(params.verbose);
    params.time_limit = extract::<f64>(params_dict, "time_limit")?.unwrap_or(params.time_limit);
    Ok(params)
}

pub fn prepare_problem<'a>(
    kind: &str,
    y: &'a &[f64],
    params: Option<&PyDict>,
) -> PyResult<Box<dyn rusvm::problem::Problem + 'a>> {
    match kind {
        "classification" => {
            check_params(
                params,
                vec!["lmbda", "smoothing", "max_asum", "shift"].as_slice(),
            )?;
            let mut problem =
                rusvm::problem::Classification::new(y, extract_params_problem(params)?);
            if let Some(shift) = extract::<f64>(params, "shift")? {
                problem.shift = shift;
            }
            Ok(Box::new(problem))
        }
        "regression" => {
            check_params(
                params,
                vec!["lmbda", "smoothing", "max_asum", "epsilon"].as_slice(),
            )?;
            let mut problem = rusvm::problem::Regression::new(y, extract_params_problem(params)?);

            if let Some(epsilon) = extract::<f64>(params, "epsilon")? {
                problem.epsilon = epsilon;
            }
            Ok(Box::new(problem))
        }
        "lssvm" => {
            check_params(params, vec!["lmbda"].as_slice())?;
            let problem = rusvm::problem::LSSVM::new(y, extract_params_problem(params)?);
            Ok(Box::new(problem))
        }
        &_ => {
            return Err(PyValueError::new_err(format!(
                "unkown problem kind '{kind}'"
            )));
        }
    }
}

pub fn prepare_callback<'py>(
    py: Python<'py>,
    callback: Option<&'py PyAny>,
) -> PyResult<Option<Box<dyn Fn(&rusvm::Status) -> bool + 'py>>> {
    let fun: Box<dyn Fn(&rusvm::Status) -> bool>;
    match callback {
        None => {
            fun = Box::new(move |_| match py.check_signals() {
                Err(_) => true,
                Ok(()) => false,
            });
            Ok(Some(fun))
        }
        Some(cb) => {
            if !cb.is_callable() {
                return Err(PyValueError::new_err("callback is not callable"));
            } else {
                fun = Box::new(move |status: &rusvm::Status| {
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
                Ok(Some(fun))
            }
        }
    }
}
