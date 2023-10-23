use numpy::ndarray::ArrayView2;

pub struct GaussianKernel<'a> {
    gamma: f64,
    data: ArrayView2<'a, f64>,
    xsqr: Vec<f64>,
}

impl<'a> GaussianKernel<'a> {
    pub fn new(gamma: f64, data: ArrayView2<'a, f64>) -> GaussianKernel<'a> {
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
    fn compute_row(&self, i: usize, ki: &mut [f64], active_set: &Vec<usize>) {
        let xsqri = self.xsqr[i];
        let xi = self.data.row(i);
        for (idx_j, &j) in active_set.iter().enumerate() {
            let xj = self.data.row(j);
            let dij = xsqri + self.xsqr[j] - 2.0 * xi.dot(&xj);
            (*ki)[idx_j] = (-self.gamma * dij).exp();
        }
    }
}

impl super::Kernel for GaussianKernel<'_> {
    fn use_rows(
        &self,
        idxs: Vec<usize>,
        active_set: &Vec<usize>,
        fun: &mut dyn FnMut(Vec<&[f64]>),
    ) {
        let mut kidxs = Vec::with_capacity(idxs.len());
        let active_size = active_set.len();
        for &idx in idxs.iter() {
            let mut kidx = vec![0.0; active_size];
            self.compute_row(idx, &mut kidx, active_set);
            kidxs.push(kidx);
        }
        fun(kidxs.iter().map(|ki| ki.as_slice()).collect());
    }

    fn diag(&self, _i: usize) -> f64 {
        1.0
    }
}
