use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

struct State {
    a: Vec<f64>,
    ka: Vec<f64>,
}

trait Problem {
    fn quad(&self, state: &State, i: usize) -> f64;
    fn grad(&self, state: &State, i: usize) -> f64;
    fn size(&self) -> usize;
}

struct Classification {
    smoothing: f64,
    lambda: f64,
    shift: f64,
    y: Vec<f64>,
    w: Vec<f64>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    sign: Vec<f64>,
}

impl Problem for Classification {
    fn quad(&self, _state: &State, i: usize) -> f64 {
        2.0 * self.smoothing * self.lambda / self.w[i]
    }
    fn grad(&self, state: &State, i: usize) -> f64 {
        state.ka[i] - self.shift * self.y[i] +
        self.smoothing * self.y[i] * (2.0 * self.y[i] * state.a[i] / self.w[i] - 1.0)
    }
    fn size(&self) -> usize {
        self.y.len()
    }
}

fn solve(problem: &dyn Problem) {
    let n = problem.size();
    let mut state = State {
        a: vec![0.0; n],
        ka: vec![0.0; n],
    };

}

/// A Python module implemented in Rust.
#[pymodule]
fn smorust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}