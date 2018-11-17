pub mod activation;
pub mod loss;

pub fn argmax(xs: &[f64]) -> usize {
    assert_ne!(xs.len(), 0);
    let mut max_value = xs[0];
    let mut max_index = 0;
    for i in 1..xs.len() {
        if max_value < xs[i] {
            max_value = xs[i];
            max_index = i;
        }
    }
    max_index
}

pub fn numerical_diff<F>(f: F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = 1e-4;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_diff_works() {
        fn function_1(x: f64) -> f64 {
            0.01 * x.powi(2) + 0.1 * x
        }
        assert_eq!(numerical_diff(function_1, 5.0), 0.1999999999990898);
        assert_eq!(numerical_diff(function_1, 10.0), 0.2999999999986347);
    }
}
