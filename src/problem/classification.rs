use crate::state::State;

pub struct Classification {
    pub smoothing: f64,
    pub lambda: f64,
    pub shift: f64,
    y: Vec<f64>,
    w: Vec<f64>,
}

impl Classification {
    pub fn new(y: Vec<f64>, lambda: f64) -> Classification {
        Classification {
            smoothing: 0.0,
            lambda,
            shift: 1.0,
            y,
            w: Vec::new(),
        }
    }

    fn weight(&self, i: usize) -> f64 {
        if self.w.len() == 0 {
            1.0
        } else {
            self.w[i]
        }
    }

    pub fn set_smoothing(mut self, smoothing: f64) -> Classification {
        self.smoothing = smoothing;
        self
    }

    pub fn set_shift(mut self, shift: f64) -> Classification {
        self.shift = shift;
        self
    }
}

impl super::Problem for Classification {
    fn quad(&self, _state: &State, i: usize) -> f64 {
        2.0 * self.smoothing * self.lambda / self.weight(i)
    }
    fn grad(&self, state: &State, i: usize) -> f64 {
        state.ka[i] - self.shift * self.y[i]
            + self.smoothing * self.y[i] * (2.0 * self.y[i] * state.a[i] / self.weight(i) - 1.0)
    }
    fn size(&self) -> usize {
        self.y.len()
    }
    fn lb(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            0.0
        } else {
            -self.weight(i)
        }
    }
    fn ub(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            self.weight(i)
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

    fn lambda(&self) -> f64 {
        self.lambda
    }
    fn regularization(&self) -> f64 {
        1e-12
    }
}
