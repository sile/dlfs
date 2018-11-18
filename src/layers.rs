use functions::activation::sigmoid;
use matrix::Matrix;

#[derive(Debug)]
pub struct ReluLayer {
    mask: Vec<bool>,
}
impl ReluLayer {
    pub fn new() -> Self {
        ReluLayer { mask: Vec::new() }
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
            .map(|(dout, y)| dout * (1.0 - y) * y)
    }
}

#[derive(Debug)]
pub struct AffineLayer {
    w: Matrix,
    b: Matrix,
    x: Matrix,
    dw: Matrix,
    db: Matrix,
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
        x.dot_product(&self.w).add(&self.b)
    }

    pub fn backward(&mut self, dout: Matrix) -> Matrix {
        let dx = dout.dot_product(&self.w.transpose());
        self.dw = self.x.transpose().dot_product(&dout);
        self.db = dout.column_sum();
        dx
    }
}
