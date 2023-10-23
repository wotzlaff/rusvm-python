pub mod classification;
use crate::{Kernel, State};
pub use classification::Classification;

pub trait Problem {
    fn quad(&self, state: &State, i: usize) -> f64;
    fn grad(&self, state: &State, i: usize) -> f64;
    fn size(&self) -> usize;
    fn lb(&self, i: usize) -> f64;
    fn ub(&self, i: usize) -> f64;
    fn sign(&self, i: usize) -> f64;
    fn is_optimal(&self, state: &State, tol: f64) -> bool;

    fn lambda(&self) -> f64;
    fn regularization(&self) -> f64;

    fn is_shrunk(&self, state: &State, active_set: &Vec<usize>) -> bool {
        active_set.len() < state.a.len()
    }

    fn shrink(
        &self,
        kernel: &mut impl Kernel,
        state: &State,
        active_set: &mut Vec<usize>,
        threshold: f64,
    ) {
        let new_active_set = active_set
            .to_vec()
            .into_iter()
            .filter(|&k| {
                let gkb = state.g[k] + state.b + state.c * self.sign(k);
                let gkb_sqr = gkb * gkb;
                gkb_sqr <= threshold * state.violation
                    || !(state.a[k] == self.ub(k) && gkb < 0.0
                        || state.a[k] == self.lb(k) && gkb > 0.0)
            })
            .collect();
        kernel.restrict_active(&active_set, &new_active_set);
        *active_set = new_active_set;
    }
    fn unshrink(&self, kernel: &mut impl Kernel, state: &mut State, active_set: &mut Vec<usize>) {
        let lambda = self.lambda();
        let n = self.size();
        let new_active_set = (0..n).collect();
        kernel.set_active(&active_set, &new_active_set);
        *active_set = new_active_set;

        state.ka.fill(0.0);
        for (i, &ai) in state.a.iter().enumerate() {
            if ai == 0.0 {
                continue;
            }
            kernel.use_rows([i].to_vec(), &active_set, &mut |ki_vec: Vec<&[f64]>| {
                let ki = ki_vec[0];
                for k in 0..n {
                    state.ka[k] += ai / lambda * ki[k];
                }
            })
        }
    }
}
