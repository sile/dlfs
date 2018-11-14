pub mod activation;
pub mod loss;

pub fn numerical_diff<F>(f: F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = 1e-4;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

pub fn numerical_gradient<F>(f: F, xs: &[f64]) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let h = 1e-4;
    let mut grad = Vec::with_capacity(xs.len());
    let mut temp = Vec::from(xs);
    for i in 0..xs.len() {
        temp[i] = xs[i] + h;
        let fxh1 = f(&temp);

        temp[i] = xs[i] - h;
        let fxh2 = f(&temp);

        grad.push((fxh1 - fxh2) / (2.0 * h));
        temp[i] = xs[i];
    }
    grad
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

    #[test]
    fn numerical_gradient_works() {
        fn function_2(xs: &[f64]) -> f64 {
            xs[0].powi(2) + xs[1].powi(2)
        }

        assert_eq!(
            numerical_gradient(function_2, &[3.0, 4.0]),
            [6.00000000000378, 7.999999999999119]
        );
    }
}
