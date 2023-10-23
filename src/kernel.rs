pub mod gaussian;
pub use gaussian::GaussianKernel;

pub trait Kernel {
    fn use_rows(&self, idxs: Vec<usize>, active_set: &Vec<usize>, fun: &mut dyn FnMut(Vec<&[f64]>));
    fn diag(&self, i: usize) -> f64;

    fn restrict_active(&mut self, _old: &Vec<usize>, _new: &Vec<usize>) {}
    fn set_active(&mut self, _old: &Vec<usize>, _new: &Vec<usize>) {}
}
