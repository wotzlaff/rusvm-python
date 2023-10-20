use numpy::ndarray::ArrayView2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
pub struct State {
    a: Vec<f64>,
    b: f64,
    violation: f64,
    value: f64,
    ka: Vec<f64>,
    g: Vec<f64>,
    active_set: Vec<usize>,
}

pub trait Problem {
    fn quad(&self, state: &State, i: usize) -> f64;
    fn grad(&self, state: &State, i: usize) -> f64;
    fn size(&self) -> usize;
    fn lb(&self, i: usize) -> f64;
    fn ub(&self, i: usize) -> f64;
    fn sign(&self, i: usize) -> f64;
    fn is_optimal(&self, state: &State, tol: f64) -> bool;

    fn get_lambda(&self) -> f64;
    fn get_regularization(&self) -> f64;
}

pub trait Kernel {
    fn use_rows(&mut self, idxs: Vec<usize>, fun: &mut dyn FnMut(Vec<&[f64]>));
    fn diag(&self, i: usize) -> f64;
}

struct GaussianKernel<'a> {
    gamma: f64,
    data: ArrayView2<'a, f64>,
    xsqr: Vec<f64>,
}

impl<'a> GaussianKernel<'a> {
    fn new(gamma: f64, data: ArrayView2<'a, f64>) -> GaussianKernel<'a> {
        let &[n, nft] = data.shape() else {
            panic!("x has bad shape");
        };
        let mut xsqr = Vec::with_capacity(n);
        for i in 0..n {
            let mut xsqri = 0.0;
            for j in 0..nft {
                xsqri += data[[i, j]] * data[[i, j]];
            }
            xsqr.push(xsqri);
        }
        GaussianKernel { gamma, data, xsqr }
    }
    fn compute_row(&mut self, i: usize, ki: &mut [f64]) {
        let &[n, _nft] = self.data.shape() else {
            panic!("x has bad shape");
        };
        let xsqri = self.xsqr[i];
        let xi = self.data.row(i);
        for j in 0..n {
            let xj = self.data.row(j);
            let dij = xsqri + self.xsqr[j] - 2.0 * xi.dot(&xj);
            (*ki)[j] = (-self.gamma * dij).exp();
        }
    }
}

impl Kernel for GaussianKernel<'_> {
    fn use_rows(&mut self, idxs: Vec<usize>, fun: &mut dyn FnMut(Vec<&[f64]>)) {
        let mut kidxs = Vec::with_capacity(2);
        for &idx in &idxs {
            // TODO: active set
            let mut kidx = vec![0.0; self.data.shape()[0]];
            self.compute_row(idx, &mut kidx);
            kidxs.push(kidx);
        }
        fun(kidxs.iter().map(|ki| ki.as_slice()).collect());
    }

    fn diag(&self, _i: usize) -> f64 {
        1.0
    }
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
        state.ka[i] - self.shift * self.y[i]
            + self.smoothing * self.y[i] * (2.0 * self.y[i] * state.a[i] / self.w[i] - 1.0)
    }
    fn size(&self) -> usize {
        self.y.len()
    }
    fn lb(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            0.0
        } else {
            -self.w[i]
        }
    }
    fn ub(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            self.w[i]
        } else {
            0.0
        }
    }
    fn sign(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    fn is_optimal(&self, state: &State, tol: f64) -> bool {
        self.lambda * state.violation < tol
    }

    fn get_lambda(&self) -> f64 {
        self.lambda
    }
    fn get_regularization(&self) -> f64 {
        1e-12
    }
}

fn find_mvp_signed(
    problem: &dyn Problem,
    state: &mut State,
    sign: f64,
) -> (f64, f64, usize, usize) {
    let mut g_min = f64::INFINITY;
    let mut g_max = f64::NEG_INFINITY;
    let mut idx_i: usize = 0;
    let mut idx_j: usize = 0;
    for (idx, &i) in state.active_set.iter().enumerate() {
        let g_i = problem.grad(state, i);
        state.g[i] = g_i;
        if problem.sign(i) * sign >= 0.0 {
            if state.a[i] > problem.lb(i) && g_i > g_max {
                idx_i = idx;
                g_max = g_i;
            }
            if state.a[i] < problem.ub(i) && g_i < g_min {
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

pub struct Result {
    a: Vec<f64>,
    b: f64,
    value: f64,
    steps: i32,
}

fn update(
    problem: &dyn Problem,
    kernel: &mut impl Kernel,
    idx_i: usize,
    idx_j: usize,
    state: &mut State,
) {
    let i = state.active_set[idx_i];
    let j = state.active_set[idx_j];
    kernel.use_rows([i, j].to_vec(), &mut |kij: Vec<&[f64]>| {
        let ki: &[f64] = kij[0];
        let kj = kij[1];
        let pij = state.g[i] - state.g[j];
        let qij = ki[idx_i] + kj[idx_j] - 2.0 * ki[idx_j]
            + problem.quad(state, i)
            + problem.quad(state, j);
        let tij = f64::min(
            problem.get_lambda() * pij / f64::max(qij, problem.get_regularization()),
            f64::min(state.a[i] - problem.lb(i), problem.ub(j) - state.a[j]),
        );
        state.a[i] -= tij;
        state.a[j] += tij;
        let tij_l = tij / problem.get_lambda();
        state.value -= tij * (0.5 * qij * tij_l - pij);
        for (idx, &k) in state.active_set.iter().enumerate() {
            state.ka[k] += tij_l * (kj[idx] - ki[idx]);
        }
    });
}

pub fn solve(
    problem: &dyn Problem,
    kernel: &mut impl Kernel,
    tol: f64,
    max_steps: i32,
    verbose: i32,
) -> Result {
    let mut result = Result {
        a: vec![],
        b: f64::NAN,
        value: 0.0,
        steps: 0,
    };
    let n = problem.size();
    let mut state = State {
        a: vec![0.0; n],
        b: 0.0,
        violation: f64::INFINITY,
        value: 0.0,
        ka: vec![0.0; n],
        g: vec![0.0; n],
        active_set: (0..n).collect(),
    };

    for step in 1..=max_steps {
        // TODO: check time limit
        // TODO: callback
        // TODO: shrinking
        let (idx_i0, idx_j1) = find_mvp(problem, &mut state);
        let optimal = problem.is_optimal(&state, tol);

        if verbose > 0 && (step % verbose == 0 || optimal) {
            println!("{:10} {:10.6} {:10.6}", step, state.violation, state.value)
        }

        if optimal {
            // TOOD: check shrinking
            result.steps = step;
            break;
        }

        // TODO: 2nd order
        let (idx_i, idx_j) = (idx_i0, idx_j1);
        update(problem, kernel, idx_i, idx_j, &mut state);
    }
    return result;
}

#[pymodule]
fn smorust<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn solve_classification<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let n = y.len();
        let problem = Classification {
            smoothing: 0.0,
            lambda: 1e-3,
            shift: 1.0,
            y: y.to_vec()?,
            w: vec![1.0; n],
        };
        let mut kernel = GaussianKernel::new(1.0, x.as_array());
        solve(&problem, &mut kernel, 1e-6, 100, 1);
        Ok(())
    }
    Ok(())
}
