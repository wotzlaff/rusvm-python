use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod prepare;
use ::rusvm::kernel::{cache, gaussian};
use prepare::*;

#[pymodule]
fn rusvm<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(signature = (x, y, params_problem = None, params_smo = None, callback = None))]
    fn solve_smo<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        params_problem: Option<&PyDict>,
        params_smo: Option<&PyDict>,
        callback: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        // get parameters
        let (params_smo, cache_size) = extract_params_smo(params_smo)?;
        // prepare problem
        let labels = y.as_slice()?;
        let problem = prepare_problem(&labels, params_problem)?;
        // prepare kernel
        let data = x.as_array();
        let base = gaussian(&data, 1.0);
        let mut kernel = cache(Box::from(base), cache_size);
        // prepare callback
        let callback = prepare_callback(py, callback)?;
        // solve problem
        let result: ::rusvm::Status = ::rusvm::smo::solve(
            problem.as_ref(),
            kernel.as_mut(),
            &params_smo,
            callback.as_deref(),
        );
        // return results
        let py_result = status_to_dict(&result, py);
        Ok(py_result)
    }

    #[pyfn(m)]
    #[pyo3(signature = (x, y, params_problem = None, params_newton = None, callback = None))]
    fn solve_newton<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        params_problem: Option<&PyDict>,
        params_newton: Option<&PyDict>,
        callback: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        // get parameters
        let (params_newton, cache_size) = extract_params_newton(params_newton)?;
        // prepare problem
        let labels = y.as_slice()?;
        let problem = prepare_problem(&labels, params_problem)?;
        // prepare kernel
        let data = x.as_array();
        let base = gaussian(&data, 1.0);
        let mut kernel = cache(Box::from(base), cache_size);
        // prepare callback
        let callback = prepare_callback(py, callback)?;
        // solve problem
        let result = ::rusvm::newton::solve(
            problem.as_ref(),
            kernel.as_mut(),
            &params_newton,
            callback.as_deref(),
        );
        // return results
        let py_result = status_to_dict(&result, py);
        Ok(py_result)
    }

    #[pyfn(m)]
    #[pyo3(signature = (x, y, params_problem = None, params_smo = None, params_newton = None, callback_smo = None, callback_newton = None))]
    fn solve_smonewt<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        params_problem: Option<&PyDict>,
        params_smo: Option<&PyDict>,
        params_newton: Option<&PyDict>,
        callback_smo: Option<&PyAny>,
        callback_newton: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        // get parameters
        let (params_smo, cache_size) = extract_params_smo(params_smo)?;
        let params_newton = extract_params_newton(params_newton)?.0;
        // prepare problem
        let labels = y.as_slice()?;
        let problem = prepare_problem(&labels, params_problem)?;
        // prepare kernel
        let data = x.as_array();
        let base = gaussian(&data, 1.0);
        let mut kernel = cache(Box::from(base), cache_size);
        // prepare callback
        let callback_smo = prepare_callback(py, callback_smo)?;
        let callback_newton = prepare_callback(py, callback_newton)?;
        // solve problem
        let params = ::rusvm::smonewt::Params {
            smo: params_smo,
            newton: params_newton,
        };
        let result = ::rusvm::smonewt::solve(
            problem.as_ref(),
            kernel.as_mut(),
            &params,
            callback_smo.as_deref(),
            callback_newton.as_deref(),
        );
        // return results
        let py_result = status_to_dict(&result, py);
        Ok(py_result)
    }

    Ok(())
}
