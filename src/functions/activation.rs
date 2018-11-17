pub fn step(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn identity(x: f64) -> f64 {
    x
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + -x.exp())
}

// Rectified Linear Unit
pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn softmax(xs: &[f64]) -> Vec<f64> {
    use std::cmp::Ordering;

    let mut max = 0.0;
    for &x in xs {
        if x.partial_cmp(&max) == Some(Ordering::Greater) {
            max = x;
        }
    }

    let mut ys = xs.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
    let sum = ys.iter().sum::<f64>();
    ys.iter_mut().for_each(|y| *y /= sum);
    ys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_works() {
        assert_eq!(
            softmax(&[1010.0, 1000.0, 990.0]),
            [
                0.999954600070331,
                0.00004539786860886666,
                0.000000002061060046209062
            ]
        );
    }
}
