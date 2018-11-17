use mnist::Mnist;
use std::mem;

use functions::activation::{sigmoid, softmax};
use functions::argmax;
use functions::loss::cross_entropy_error;
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
        let w1 = Matrix::with_randn(input_size, hidden_size) * options.weight_init_std;
        let b1 = Matrix::from(vec![vec![0f64; hidden_size]]);
        let w2 = Matrix::with_randn(hidden_size, output_size) * options.weight_init_std;
        let b2 = Matrix::from(vec![vec![0f64; output_size]]);
        TwoLayerNet { w1, b1, w2, b2 }
    }

    pub fn predict(&self, xs: &[f64]) -> Vec<f64> {
        let xs = Matrix::from(vec![Vec::from(xs)]);
        let a1 = xs.dot_product(&self.w1).add(&self.b1);
        let z1 = a1.map(|&v| sigmoid(v));
        let a2 = z1.dot_product(&self.w2).add(&self.b2);
        let ys = a2.map_row(softmax).into_vector();
        ys
    }

    pub fn loss(&self, data: &[f64], labels: &[f64]) -> f64 {
        let ys = self.predict(data);
        cross_entropy_error(&ys, labels)
    }

    pub fn batched_loss<'a, I, J>(&self, batched_data: I, batched_labels: J) -> f64
    where
        I: Iterator<Item = &'a [f64]>,
        J: Iterator<Item = &'a [f64]>,
    {
        let mut sum = 0.0;
        let mut len = 0;
        for (data, labels) in batched_data.zip(batched_labels) {
            sum += self.loss(data, labels);
            len += 1;
        }
        sum / (len as f64)
    }

    pub fn accuracy(&self, data: &[f64], labels: &[f64]) -> f64 {
        let ys = self.predict(data);
        let y = argmax(&ys);
        let t = argmax(&labels);
        if y == t {
            1.0
        } else {
            0.0
        }
    }

    pub fn numerical_gradient(mut self, data: &[f64], labels: &[f64]) -> Gradients {
        let w1 = self.w1.clone().numerical_gradient(|m| {
            let old = mem::replace(&mut self.w1, m.clone());
            let loss = self.loss(data, labels);
            self.w1 = old;
            loss
        });
        let b1 = self.w1.clone().numerical_gradient(|m| {
            let old = mem::replace(&mut self.b1, m.clone());
            let loss = self.loss(data, labels);
            self.b1 = old;
            loss
        });
        let w2 = self.w2.clone().numerical_gradient(|m| {
            let old = mem::replace(&mut self.w2, m.clone());
            let loss = self.loss(data, labels);
            self.w2 = old;
            loss
        });
        let b2 = self.w2.clone().numerical_gradient(|m| {
            let old = mem::replace(&mut self.b2, m.clone());
            let loss = self.loss(data, labels);
            self.b2 = old;
            loss
        });
        Gradients { w1, b1, w2, b2 }
    }

    pub fn batched_numerical_gradient<'a, I, J>(
        mut self,
        batched_data: &'a I,
        batched_labels: &'a J,
    ) -> Gradients
    where
        I: Iterator<Item = &'a [f64]> + Clone,
        J: Iterator<Item = &'a [f64]> + Clone,
    {
        let w1 = self.w1.clone().numerical_gradient(|m| {
            let old = mem::replace(&mut self.w1, m.clone());
            let loss = self.batched_loss(batched_data.clone(), batched_labels.clone());
            self.w1 = old;
            loss
        });
        let b1 = self.w1.clone().numerical_gradient(|m| {
            let old = mem::replace(&mut self.b1, m.clone());
            let loss = self.batched_loss(batched_data.clone(), batched_labels.clone());
            self.b1 = old;
            loss
        });
        let w2 = self.w2.clone().numerical_gradient(|m| {
            let old = mem::replace(&mut self.w2, m.clone());
            let loss = self.batched_loss(batched_data.clone(), batched_labels.clone());
            self.w2 = old;
            loss
        });
        let b2 = self.w2.clone().numerical_gradient(|m| {
            let old = mem::replace(&mut self.b2, m.clone());
            let loss = self.batched_loss(batched_data.clone(), batched_labels.clone());
            self.b2 = old;
            loss
        });
        Gradients { w1, b1, w2, b2 }
    }

    pub fn train(&self, mnist: &Mnist) {}
}

#[derive(Debug)]
pub struct Gradients {
    pub w1: Matrix,
    pub b1: Matrix,
    pub w2: Matrix,
    pub b2: Matrix,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_layer_net_works() {
        let net = TwoLayerNet::new(784, 100, 10, &Default::default());
        assert_eq!(net.w1.shape(), (784, 100));
        assert_eq!(net.b1.shape(), (1, 100));
        assert_eq!(net.w2.shape(), (100, 10));
        assert_eq!(net.b2.shape(), (1, 10));
    }
}
