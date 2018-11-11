use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};

// FIXME: optimize inner representation
//        (e.g., `{ inner: Vec<T>, rows: usize, cols: usize }`)
#[derive(Debug, PartialEq, Eq)]
pub struct Matrix<T>(Vec<Vec<T>>);
impl<T> Matrix<T>
where
    T: Default + Clone,
{
    pub fn new(rows: usize, columns: usize) -> Self {
        Matrix((0..rows).map(|_| vec![T::default(); columns]).collect())
    }
}
impl<T> Matrix<T>
where
    T: Default + Clone + Mul<Output = T> + Add + Sub + Div + Sum,
{
    pub fn dot_product(&self, other: &Self) -> Self {
        assert_eq!(self.rows(), other.columns());
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

    pub fn rows(&self) -> usize {
        self.0.get(0).map_or(0, |x| x.len())
    }

    pub fn columns(&self) -> usize {
        self.0.len()
    }

    pub fn row(&self, i: usize) -> impl Iterator<Item = &T> {
        // FIXME: Make this safe
        self.0[i].iter()
    }

    pub fn column(&self, i: usize) -> impl Iterator<Item = &T> {
        // FIXME: Make this safe
        (0..self.rows()).map(move |j| &self.0[j][i])
    }

    pub fn into_vec(self) -> Vec<Vec<T>> {
        self.0
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
}
