use data::Mnist;
use functions::argmax;
use layers::{AffineLayer, ReluLayer, SoftmaxWithLossLayer};
use matrix::Matrix;

#[derive(Debug)]
pub struct TwoLayerNet {
    affine1_layer: AffineLayer,
    relu1_layer: ReluLayer,
    affine2_layer: AffineLayer,
    last_layer: SoftmaxWithLossLayer,
}
impl TwoLayerNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let w1 = Matrix::with_randn(input_size, hidden_size);
        let b1 = Matrix::from(vec![vec![0f64; hidden_size]]);
        let w2 = Matrix::with_randn(hidden_size, output_size);
        let b2 = Matrix::from(vec![vec![0f64; output_size]]);

        let affine1_layer = AffineLayer::new(w1, b1);
        let relu1_layer = ReluLayer::new();
        let affine2_layer = AffineLayer::new(w2, b2);
        let last_layer = SoftmaxWithLossLayer::new();
        TwoLayerNet {
            affine1_layer,
            relu1_layer,
            affine2_layer,
            last_layer,
        }
    }

    pub fn predict(&mut self, xs: Matrix) -> Matrix {
        let xs = self.affine1_layer.forward(xs);
        let xs = self.relu1_layer.forward(xs);
        let xs = self.affine2_layer.forward(xs);
        xs
    }

    pub fn loss(&mut self, x: Matrix, t: Matrix) -> Vec<f64> {
        let y = self.predict(x);
        self.last_layer.forward(y, t)
    }

    pub fn accuracy(&mut self, x: Matrix, t: Matrix) -> f64 {
        let y = self.predict(x);
        let mut oks = 0;
        for i in 0..y.rows() {
            if argmax(y.row_slice(i)) == argmax(t.row_slice(i)) {
                oks += 1;
            }
        }
        (oks as f64) / (t.rows() as f64)
    }

    pub fn gradient(&mut self, x: Matrix, t: Matrix) -> Gradients {
        // forward
        self.loss(x, t);

        // backword
        let dout = self.last_layer.backword();
        let dout = self.affine2_layer.backward(dout);
        let dout = self.relu1_layer.backward(dout);
        let _dout = self.affine1_layer.backward(dout);

        let w1 = self.affine1_layer.dw.clone();
        let b1 = self.affine1_layer.db.clone();
        let w2 = self.affine2_layer.dw.clone();
        let b2 = self.affine2_layer.db.clone();
        Gradients { w1, b1, w2, b2 }
    }

    pub fn train(
        &mut self,
        mnist: &Mnist,
        iters_num: usize,
        batch_size: usize,
        learning_rate: f64,
    ) {
        println!("# START: TRAIN");
        for i in 0..iters_num {
            let (x_batch, t_batch) = mnist.choice_train_batch2(batch_size);
            let grad = self.gradient(x_batch.clone(), t_batch.clone());
            self.affine1_layer.w -= grad.w1 * learning_rate;
            self.affine1_layer.b -= grad.b1 * learning_rate;
            self.affine2_layer.w -= grad.w2 * learning_rate;
            self.affine2_layer.b -= grad.b2 * learning_rate;

            let loss = self.loss(x_batch, t_batch);
            println!("[{}/{}]: LOSS={:?}", i + 1, iters_num, loss);
        }
        println!(" END: TRAIN");
    }
}
impl Default for TwoLayerNet {
    fn default() -> Self {
        TwoLayerNet::new(784, 50, 10)
    }
}

#[derive(Debug)]
pub struct Gradients {
    pub w1: Matrix,
    pub b1: Matrix,
    pub w2: Matrix,
    pub b2: Matrix,
}
