use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod prepare;
use prepare::*;

#[pymodule]
fn pyrusvm<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
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
        // get SMO parameter
        let params_smo = extract_params_smo(params_smo)?;
        // prepare problem
        let y = y.as_slice()?;
        let problem = prepare_problem(kind, &y, params_problem)?;
        // prepare kernel
        let mut base = Box::new(rusvm::kernel::GaussianKernel::new(1.0, x.as_array()));
        let mut kernel: Box<dyn rusvm::kernel::Kernel> = {
            if cache_size > 0 {
                Box::new(rusvm::kernel::CachedKernel::from(base.as_mut(), cache_size))
            } else {
                base
            }
        };
        // prepare callback
        let callback = prepare_callback(py, callback)?;
        // solve problem
        let result = rusvm::smo::solve(
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
    #[pyo3(signature = (x, y, kind = "classification", params_problem = None, params_newton = None, cache_size = 0, callback = None))]
    fn solve_newton<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        kind: &str,
        params_problem: Option<&PyDict>,
        params_newton: Option<&PyDict>,
        cache_size: usize,
        callback: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        // get SMO parameter
        let params_newton = extract_params_newton(params_newton)?;
        // prepare problem
        let y = y.as_slice()?;
        let problem = prepare_problem(kind, &y, params_problem)?;
        // prepare kernel
        let mut base = Box::new(rusvm::kernel::GaussianKernel::new(1.0, x.as_array()));
        let mut kernel: Box<dyn rusvm::kernel::Kernel> = {
            if cache_size > 0 {
                Box::new(rusvm::kernel::CachedKernel::from(base.as_mut(), cache_size))
            } else {
                base
            }
        };
        // prepare callback
        let callback = prepare_callback(py, callback)?;
        // solve problem
        let result = rusvm::newton::solve(
            problem.as_ref(),
            kernel.as_mut(),
            &params_newton,
            callback.as_deref(),
        );
        // return results
        let py_result = status_to_dict(&result, py);
        Ok(py_result)
    }

    Ok(())
}
