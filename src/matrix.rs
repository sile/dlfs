use rand::distributions::StandardNormal;
use rand::rngs::StdRng;
use rand::{FromEntropy, Rng};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use image::Image;

// FIXME: optimize inner representation
//        (e.g., `{ inner: Vec<T>, rows: usize, cols: usize }`)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T = f64>(Vec<Vec<T>>);
impl<T> Matrix<T>
where
    T: Default + Clone,
{
    pub fn new(rows: usize, columns: usize) -> Self {
        Matrix((0..rows).map(|_| vec![T::default(); columns]).collect())
    }
}
impl Matrix<f64> {
    pub fn with_randn(rows: usize, columns: usize) -> Self {
        let mut m = Self::new(rows, columns);
        let mut rng = StdRng::from_entropy();
        for row in m.0.iter_mut() {
            for cell in row.iter_mut() {
                *cell = rng.sample(&StandardNormal);
            }
        }
        m
    }

    // a.k.a., img2col
    //
    // image: (batch_size, channel, height, width)
    pub fn from_images<I>(
        images: I,
        filter_h: usize,
        filter_w: usize,
        stride: usize,
        pad: usize,
    ) -> Self
    where
        I: Iterator<Item = Image>,
    {
        let filter_h = filter_h as isize;
        let filter_w = filter_w as isize;
        let pad = pad as isize;

        let mut rows = Vec::new();
        for image in images {
            let height = image.height() as isize;
            let width = image.width() as isize;

            let y_start = -(pad as isize);
            let y_end = (height + pad) as isize;
            let x_start = -(pad as isize);
            let x_end = (width + pad) as isize;
            for y in y_start..(y_end - filter_h) + 1 {
                if (y - y_start) % (stride as isize) != 0 {
                    continue;
                }

                for x in x_start..(x_end - filter_w) + 1 {
                    if (x - x_start) % (stride as isize) != 0 {
                        continue;
                    }

                    let mut row = Vec::new();
                    for image_per_channel in &image.0 {
                        for i in y..y + filter_h {
                            for j in x..x + filter_w {
                                if i < 0 || j < 0 || i >= height || j >= width {
                                    row.push(0.0);
                                    continue;
                                }
                                row.push(image_per_channel[i as usize][j as usize]);
                            }
                        }
                    }
                    rows.push(row);
                }
            }
        }
        Matrix::from(rows)
    }

    pub fn column_sum(&self) -> Matrix {
        let mut m = Matrix::new(1, self.columns());
        for i in 0..self.columns() {
            m.0[0][i] = self.column(i).sum();
        }
        m
    }

    pub fn transpose(&self) -> Matrix {
        let mut m = Matrix::new(self.columns(), self.rows());
        for y in 0..self.rows() {
            for x in 0..self.columns() {
                m.0[x][y] = self.0[y][x];
            }
        }
        m
    }

    pub fn numerical_gradient<F>(&mut self, mut f: F) -> Matrix<f64>
    where
        F: FnMut(&Self) -> f64,
    {
        let h = 1e-4;
        let mut grad = Matrix::new(self.rows(), self.columns());
        for y in 0..self.rows() {
            for x in 0..self.columns() {
                let temp = self.0[y][x];
                self.0[y][x] = temp + h;
                let fxh1 = f(self);

                self.0[y][x] = temp - h;
                let fxh2 = f(self);

                grad.0[y][x] = (fxh1 - fxh2) / (2.0 * h);
                self.0[y][x] = temp;
            }
        }
        grad
    }
}
impl<T> Matrix<T>
where
    T: Default
        + Clone
        + Mul<Output = T>
        + Add
        + AddAssign
        + Sub
        + SubAssign
        + Div
        + Sum
        + std::fmt::Debug,
{
    pub fn dot_product(&self, other: &Self) -> Self {
        assert_eq!(self.columns(), other.rows());
        let mut out = Matrix::new(self.rows(), other.columns());
        for y in 0..self.rows() {
            for x in 0..other.columns() {
                out.0[y][x] = self
                    .row(y)
                    .cloned()
                    .zip(other.column(x).cloned())
                    .map(|t| t.0 * t.1)
                    .sum();
            }
        }
        out
    }

    pub fn add(mut self, other: &Self) -> Self {
        assert_eq!(
            self.rows(),
            other.rows(),
            "self={:?}, rhs={:?}",
            self.shape(),
            other.shape()
        );
        assert_eq!(
            self.columns(),
            other.columns(),
            "self={:?}, rhs={:?}",
            self.shape(),
            other.shape()
        );
        for y in 0..self.rows() {
            for x in 0..self.columns() {
                self.0[y][x] += other.0[y][x].clone();
            }
        }
        self
    }

    pub fn add_vector(mut self, other: &Self) -> Self {
        assert_eq!(other.rows(), 1);
        assert_eq!(
            self.columns(),
            other.columns(),
            "self={:?}, rhs={:?}",
            self.shape(),
            other.shape()
        );
        for y in 0..self.rows() {
            for x in 0..self.columns() {
                self.0[y][x] += other.0[0][x].clone();
            }
        }
        self
    }

    pub fn sub(mut self, other: &Self) -> Self {
        assert_eq!(
            self.rows(),
            other.rows(),
            "self={:?}, rhs={:?}",
            self.shape(),
            other.shape()
        );
        assert_eq!(
            self.columns(),
            other.columns(),
            "self={:?}, rhs={:?}",
            self.shape(),
            other.shape()
        );
        for y in 0..self.rows() {
            for x in 0..self.columns() {
                self.0[y][x] -= other.0[y][x].clone();
            }
        }
        self
    }

    pub fn rows(&self) -> usize {
        self.0.len()
    }

    pub fn columns(&self) -> usize {
        self.0.get(0).map_or(0, |x| x.len())
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.columns())
    }

    pub fn row(&self, i: usize) -> impl Iterator<Item = &T> {
        // FIXME: Make this safe
        self.0[i].iter()
    }

    pub fn row_slice(&self, i: usize) -> &[T] {
        // FIXME: Make this safe
        &self.0[i]
    }

    pub fn column(&self, i: usize) -> impl Iterator<Item = &T> {
        // FIXME: Make this safe
        (0..self.rows()).map(move |j| &self.0[j][i])
    }

    pub fn into_vec(self) -> Vec<Vec<T>> {
        self.0
    }

    pub fn into_vector(mut self) -> Vec<T> {
        assert_eq!(self.rows(), 1);
        self.0.swap_remove(0)
    }

    pub fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(&T) -> T,
    {
        for y in 0..self.rows() {
            for x in 0..self.columns() {
                let t = f(&self.0[y][x]);
                self.0[y][x] = t;
            }
        }
        self
    }

    pub fn map_row<F>(self, f: F) -> Self
    where
        F: Fn(&[T]) -> Vec<T>,
    {
        let m = self.0.into_iter().map(|row| f(&row)).collect();
        Matrix(m)
    }
}
impl Matrix {
    pub fn sqrt(&self) -> Matrix {
        let mut m = self.clone();
        for row in m.0.iter_mut() {
            for cell in row.iter_mut() {
                *cell = cell.sqrt();
            }
        }
        m
    }
}
impl<T> From<Vec<Vec<T>>> for Matrix<T> {
    fn from(f: Vec<Vec<T>>) -> Self {
        let columns = f.get(0).map_or(0, |x| x.len());
        for row in &f {
            // FIXME
            assert_eq!(row.len(), columns);
        }
        Matrix(f)
    }
}
impl Mul<f64> for Matrix<f64> {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self {
        for row in self.0.iter_mut() {
            for cell in row.iter_mut() {
                *cell *= rhs;
            }
        }
        self
    }
}
impl Mul<Matrix> for Matrix {
    type Output = Self;
    fn mul(mut self, rhs: Matrix) -> Self {
        for y in 0..self.rows() {
            for x in 0..self.columns() {
                self.0[y][x] *= rhs.0[y][x];
            }
        }
        self
    }
}
impl Div<f64> for Matrix<f64> {
    type Output = Self;
    fn div(mut self, rhs: f64) -> Self {
        for row in self.0.iter_mut() {
            for cell in row.iter_mut() {
                *cell /= rhs;
            }
        }
        self
    }
}
impl Div<Matrix> for Matrix {
    type Output = Self;
    fn div(mut self, rhs: Matrix) -> Self {
        for y in 0..self.rows() {
            for x in 0..self.columns() {
                self.0[y][x] /= rhs.0[y][x];
            }
        }
        self
    }
}
impl Add<f64> for Matrix<f64> {
    type Output = Self;
    fn add(mut self, rhs: f64) -> Self {
        for row in self.0.iter_mut() {
            for cell in row.iter_mut() {
                *cell += rhs;
            }
        }
        self
    }
}
impl SubAssign for Matrix<f64> {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows(), rhs.rows());
        assert_eq!(self.columns(), rhs.columns());
        for y in 0..self.rows() {
            for x in 0..self.columns() {
                self.0[y][x] -= rhs.0[y][x];
            }
        }
    }
}
impl AddAssign for Matrix<f64> {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows(), rhs.rows());
        assert_eq!(self.columns(), rhs.columns());
        for y in 0..self.rows() {
            for x in 0..self.columns() {
                self.0[y][x] += rhs.0[y][x];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = Matrix::from(vec![vec![1, 2], vec![3, 4]]);
        let b = Matrix::from(vec![vec![5, 6], vec![7, 8]]);

        assert_eq!(
            a.dot_product(&b).into_vec(),
            vec![vec![19, 22], vec![43, 50]]
        );
    }

    #[test]
    fn section_3_4_3() {
        use functions::activation::{identity, sigmoid};

        struct Network {
            w1: Matrix,
            b1: Matrix,
            w2: Matrix,
            b2: Matrix,
            w3: Matrix,
            b3: Matrix,
        }
        impl Network {
            fn forward(&self, input: &[f64]) -> Vec<f64> {
                let x = Matrix::from(vec![Vec::from(input)]);
                let a1 = x.dot_product(&self.w1).add(&self.b1);
                let z1 = a1.map(|&v| sigmoid(v));
                let a2 = z1.dot_product(&self.w2).add(&self.b2);
                let z2 = a2.map(|&v| sigmoid(v));
                let a3 = z2.dot_product(&self.w3).add(&self.b3);
                let y = a3.map(|&v| identity(v));
                y.into_vec().swap_remove(0)
            }
        }

        let network = Network {
            w1: Matrix::from(vec![vec![0.0, 0.3, 0.5], vec![0.2, 0.4, 0.6]]),
            b1: Matrix::from(vec![vec![0.1, 0.2, 0.3]]),
            w2: Matrix::from(vec![vec![0.1, 0.4], vec![0.2, 0.5], vec![0.3, 0.6]]),
            b2: Matrix::from(vec![vec![0.1, 0.2]]),
            w3: Matrix::from(vec![vec![0.1, 0.3], vec![0.2, 0.4]]),
            b3: Matrix::from(vec![vec![0.1, 0.2]]),
        };

        let x = [1.0, 0.5];
        let y = network.forward(&x);
        assert_eq!(y, [0.3164209556565184, 0.6954092315109959]);
    }

    #[test]
    fn im2col() {
        use std::iter::{once, repeat};

        let image = Image(vec![vec![vec![0.0; 7]; 7]; 3]);
        let m = Matrix::from_images(once(image.clone()), 5, 5, 1, 0);
        assert_eq!(m.shape(), (9, 75));

        let m = Matrix::from_images(repeat(image).take(10), 5, 5, 1, 0);
        assert_eq!(m.shape(), (90, 75));
    }
}
