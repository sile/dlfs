use std::f64::EPSILON;

pub fn mean_squared_error(predicted: &[f64], observed: &[f64]) -> f64 {
    0.5 * predicted
        .iter()
        .zip(observed.iter())
        .map(|t| (t.0 - t.1).powi(2))
        .sum::<f64>()
}

pub fn cross_entropy_error(predicted: &[f64], observed: &[f64]) -> f64 {
    -predicted
        .iter()
        .zip(observed.iter())
        .map(|t| *t.1 * (if *t.0 == 0.0 { EPSILON } else { *t.0 }).ln())
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_squared_error_works() {
        let observed = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let predicted = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0];
        assert_eq!(
            mean_squared_error(&predicted[..], &observed[..]),
            0.09750000000000003
        );
    }

    #[test]
    fn cross_entropy_error_works() {
        let observed = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let predicted = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0];
        assert_eq!(
            cross_entropy_error(&predicted[..], &observed[..]),
            0.5108256237659907
        );
    }
}
