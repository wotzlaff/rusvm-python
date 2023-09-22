use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

pub struct State {
    a: Vec<f64>,
    b: f64,
    violation: f64,
    ka: Vec<f64>,
    g: Vec<f64>,
    active_set: Vec<usize>,
    // d_dn: Vec<f64>,
    // d_up: Vec<f64>,
}

pub trait Problem {
    fn quad(&self, state: &State, i: usize) -> f64;
    fn grad(&self, state: &State, i: usize) -> f64;
    fn size(&self) -> usize;
    fn lb(&self, i: usize) -> f64;
    fn ub(&self, i: usize) -> f64;
    fn sign(&self, i: usize) -> f64;
}

pub struct Classification {
    smoothing: f64,
    lambda: f64,
    shift: f64,
    y: Vec<f64>,
    w: Vec<f64>,
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
    fn lb(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 { 0.0 } else { -self.w[i] }
    }
    fn ub(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 { self.w[i] } else { 0.0 }
    }
    fn sign(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 { 1.0 } else {-1.0}
    }
}

fn find_mvp_signed(problem: &dyn Problem, state: &mut State, sign: f64) -> (f64, f64, usize, usize) {
    let mut g_min = f64::INFINITY;
    let mut g_max = f64::NEG_INFINITY;
    let mut idx_i: usize = 0;
    let mut idx_j: usize = 0;
    for (idx, i) in state.active_set.iter().enumerate() {
        let g_i = problem.grad(state, *i);
        state.g[*i] = g_i;
        if problem.sign(*i) * sign >= 0.0 {
            if state.a[*i] > problem.lb(*i) && g_i > g_max {
                idx_i = idx;
                g_max = g_i;
            }
            if state.a[*i] < problem.ub(*i) && g_i < g_min {
                idx_j = idx;
                g_min = g_i;
            }
        }
    }
    (g_max - g_min, g_max + g_min, idx_i, idx_j)
}

fn find_mvp(problem: &dyn Problem, state: &mut State) -> (usize, usize) {
    let (dij, sij, idx_i, idx_j) = find_mvp_signed(problem, state, 0.0);
    state.b = -0.5 * sij;
    state.violation = dij;
    (idx_i, idx_j)
}

pub fn solve(
    problem: &dyn Problem,
    tol: f64,
    max_steps: i32,
) {
    let n = problem.size();
    let mut state = State {
        a: vec![0.0; n],
        b: 0.0,
        violation: f64::INFINITY,
        ka: vec![0.0; n],
        g: vec![0.0; n],
        active_set: (0..n).collect(),
    };

    for step in 1..=max_steps {
        let (idx_i0, idx_j1) = find_mvp(problem, &mut state);
        let optimal = problem.lambda * state.violation < tol;
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn smorust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}