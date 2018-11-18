use mnist;
use rand;
use std::path::Path;

use matrix::Matrix;

const IMAGE_SIZE: usize = 28 * 28;

#[derive(Debug)]
pub struct Mnist {
    x_train: Vec<f64>,
    y_train: Vec<f64>,
}
impl Mnist {
    pub fn load<P: AsRef<Path>>(data_dir: P) -> Self {
        let mnist = mnist::MnistBuilder::new()
            .base_path(data_dir.as_ref().to_str().expect("Wrong path"))
            .label_format_one_hot()
            .finalize();
        let x_train = mnist
            .trn_img
            .into_iter()
            .map(|v| (v as f64) / 255.0)
            .collect();
        let y_train = mnist.trn_lbl.into_iter().map(|v| v as f64).collect();
        Mnist { x_train, y_train }
    }

    pub fn train_image_count(&self) -> usize {
        self.x_train.len() / IMAGE_SIZE
    }

    pub fn train_label_count(&self) -> usize {
        self.y_train.len() / 10
    }

    pub fn train_image(&self, index: usize) -> &[f64] {
        &self.x_train[index * IMAGE_SIZE..][..IMAGE_SIZE]
    }

    pub fn train_label(&self, index: usize) -> &[f64] {
        &self.y_train[index * 10..][..10]
    }

    pub fn choice_train_batch(&self, batch_size: usize) -> impl Iterator<Item = MnistEntry> {
        (0..batch_size).map(move |_| {
            let i = rand::random::<usize>() % self.train_image_count();
            MnistEntry {
                image: self.train_image(i),
                label: self.train_label(i),
            }
        })
    }

    pub fn choice_train_batch2(&self, batch_size: usize) -> (Matrix, Matrix) {
        let mut x_batch = Vec::new();
        let mut y_batch = Vec::new();
        for t in self.choice_train_batch(batch_size) {
            x_batch.push(Vec::from(t.image));
            y_batch.push(Vec::from(t.label));
        }
        (Matrix::from(x_batch), Matrix::from(y_batch))
    }
}

#[derive(Debug, Clone)]
pub struct MnistEntry<'a> {
    pub image: &'a [f64],
    pub label: &'a [f64],
}
