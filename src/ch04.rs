use matrix::Matrix;

#[derive(Debug, Clone)]
pub struct TwoLayerNetOptions {
    pub weight_init_std: f64,
}
impl Default for TwoLayerNetOptions {
    fn default() -> Self {
        TwoLayerNetOptions {
            weight_init_std: 0.01,
        }
    }
}

#[derive(Debug)]
pub struct TwoLayerNet {
    w1: Matrix,
    b1: Matrix,
    w2: Matrix,
    b2: Matrix,
}
impl TwoLayerNet {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        options: &TwoLayerNetOptions,
    ) -> Self {
        let b1 = Matrix::from(vec![vec![0; hidden_size]]);
        let b2 = Matrix::from(vec![vec![0; output_size]]);
        panic!()
    }
}
