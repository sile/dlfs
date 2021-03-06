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

pub fn gradient_descent<F>(f: F, xs: &mut [f64], learning_rate: f64, step_num: usize)
where
    F: Fn(&[f64]) -> f64,
{
    for _ in 0..step_num {
        let grads = numerical_gradient(&f, xs);
        for (x, grad) in xs.iter_mut().zip(grads.into_iter()) {
            *x -= learning_rate * grad;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn gradient_descent_works() {
        fn function_2(xs: &[f64]) -> f64 {
            xs[0].powi(2) + xs[1].powi(2)
        }

        let mut xs = [-3.0, 4.0];
        gradient_descent(function_2, &mut xs, 0.1, 100);
        assert_eq!(
            xs,
            [-0.0000000006111107928998789, 0.0000000008148143905314271]
        );
    }
}
