// [channel][height][width]
#[derive(Debug, Clone)]
pub struct Image(pub Vec<Vec<Vec<f64>>>);
impl Image {
    pub fn channels(&self) -> usize {
        self.0.len()
    }

    pub fn height(&self) -> usize {
        self.0[0].len()
    }

    pub fn width(&self) -> usize {
        self.0[0][0].len()
    }
}
