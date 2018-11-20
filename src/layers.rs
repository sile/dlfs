use functions::activation::{sigmoid, softmax};
use functions::loss::cross_entropy_error;
use matrix::Matrix;

#[derive(Debug)]
pub struct ReluLayer {
    rows: Vec<ReluLayerInner>,
}
impl ReluLayer {
    pub fn new() -> Self {
        ReluLayer { rows: Vec::new() }
    }

    pub fn forward(&mut self, xs: Matrix) -> Matrix {
        let mut out = Vec::new();

        self.rows.clear();
        for i in 0..xs.rows() {
            let mut relu = ReluLayerInner::new();
            out.push(relu.forward(xs.row(i).cloned()).collect::<Vec<_>>());
            self.rows.push(relu);
        }
        Matrix::from(out)
    }

    pub fn backward(&mut self, dout: Matrix) -> Matrix {
        let mut dx = Vec::new();
        for y in 0..dout.rows() {
            dx.push(
                self.rows[y]
                    .backward(dout.row(y).cloned())
                    .collect::<Vec<_>>(),
            );
        }
        Matrix::from(dx)
    }
}

#[derive(Debug)]
struct ReluLayerInner {
    mask: Vec<bool>,
}
impl ReluLayerInner {
    pub fn new() -> Self {
        ReluLayerInner { mask: Vec::new() }
    }

    pub fn forward<'a, I>(&'a mut self, xs: I) -> impl 'a + Iterator<Item = f64>
    where
        I: 'a + Iterator<Item = f64>,
    {
        self.mask.clear();
        xs.map(move |x| {
            if x <= 0.0 {
                self.mask.push(true);
                0.0
            } else {
                self.mask.push(false);
                x
            }
        })
    }

    pub fn backward<'a, I>(&'a self, dout: I) -> impl 'a + Iterator<Item = f64>
    where
        I: 'a + Iterator<Item = f64>,
    {
        self.mask
            .iter()
            .zip(dout)
            .map(|(&mask, x)| if mask { 0.0 } else { x })
    }
}

#[derive(Debug)]
pub struct SigmoidLayer {
    out: Vec<f64>,
}
impl SigmoidLayer {
    pub fn new() -> Self {
        SigmoidLayer { out: Vec::new() }
    }

    pub fn forward<'a, I>(&'a mut self, xs: I) -> impl 'a + Iterator<Item = f64>
    where
        I: 'a + Iterator<Item = f64>,
    {
        self.out.clear();
        xs.map(move |x| {
            let y = sigmoid(x);
            self.out.push(x);
            y
        })
    }

    pub fn backword<'a, I>(&'a mut self, dout: I) -> impl 'a + Iterator<Item = f64>
    where
        I: 'a + Iterator<Item = f64>,
    {
        self.out
            .iter()
            .zip(dout)
            .map(|(y, dout)| dout * (1.0 - y) * y)
    }
}

#[derive(Debug)]
pub struct AffineLayer {
    pub w: Matrix,
    pub b: Matrix,
    x: Matrix,
    pub dw: Matrix,
    pub db: Matrix,
}
impl AffineLayer {
    pub fn new(w: Matrix, b: Matrix) -> Self {
        AffineLayer {
            w,
            b,
            x: Matrix::new(0, 0),
            dw: Matrix::new(0, 0),
            db: Matrix::new(0, 0),
        }
    }

    pub fn forward(&mut self, x: Matrix) -> Matrix {
        self.x = x.clone();
        x.dot_product(&self.w).add_vector(&self.b)
    }

    pub fn backward(&mut self, dout: Matrix) -> Matrix {
        let dx = dout.dot_product(&self.w.transpose());
        self.dw = self.x.transpose().dot_product(&dout);
        self.db = dout.column_sum();
        dx
    }
}

#[derive(Debug)]
pub struct SoftmaxWithLossLayer {
    y: Matrix, // softmaxの出力
    t: Matrix, // 教師データ (one-hot vector)
}
impl SoftmaxWithLossLayer {
    pub fn new() -> Self {
        SoftmaxWithLossLayer {
            y: Matrix::new(0, 0),
            t: Matrix::new(0, 0),
        }
    }

    pub fn forward(&mut self, x: Matrix, t: Matrix) -> Vec<f64> {
        self.t = t.clone();
        self.y = x.clone().map_row(softmax);

        let mut loss = Vec::new();
        assert_eq!(self.y.rows(), t.rows());
        for i in 0..self.y.rows() {
            loss.push(cross_entropy_error(self.y.row_slice(i), t.row_slice(i)));
        }
        loss
    }

    pub fn backword(&mut self) -> Matrix {
        let batch_size = self.t.rows();

        // NOTE: `AffineLayer`でバッチサイズ分の合算が行われるので、ここであらかじめ`batch_size`で割って単位を合わせておく
        let dx = self.y.clone().sub(&self.t) / (batch_size as f64);
        dx
    }
}
