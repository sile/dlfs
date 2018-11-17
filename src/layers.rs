use functions::activation::sigmoid;

#[derive(Debug)]
pub struct ReluLayer {
    mask: Vec<bool>,
}
impl ReluLayer {
    pub fn new() -> Self {
        ReluLayer { mask: Vec::new() }
    }

    pub fn forward<'a, I>(&'a mut self, xs: I) -> impl 'a + Iterator<Item = f64>
    where
        I: 'a + Iterator<Item = f64>,
    {
        self.mask.clear();
        xs.map(move |x| {
            if x <= 0.0 {
                self.mask.push(true);
                0.0
            } else {
                self.mask.push(false);
                x
            }
        })
    }

    pub fn backward<'a, I>(&'a self, dout: I) -> impl 'a + Iterator<Item = f64>
    where
        I: 'a + Iterator<Item = f64>,
    {
        self.mask
            .iter()
            .zip(dout)
            .map(|(&mask, x)| if mask { 0.0 } else { x })
    }
}

#[derive(Debug)]
pub struct SigmoidLayer {
    out: Vec<f64>,
}
impl SigmoidLayer {
    pub fn new() -> Self {
        SigmoidLayer { out: Vec::new() }
    }

    pub fn forward<'a, I>(&'a mut self, xs: I) -> impl 'a + Iterator<Item = f64>
    where
        I: 'a + Iterator<Item = f64>,
    {
        self.out.clear();
        xs.map(move |x| {
            let y = sigmoid(x);
            self.out.push(x);
            y
        })
    }

    pub fn backword<'a, I>(&'a mut self, dout: I) -> impl 'a + Iterator<Item = f64>
    where
        I: 'a + Iterator<Item = f64>,
    {
        self.out
            .iter()
            .zip(dout)
            .map(|(dout, y)| dout * (1.0 - y) * y)
    }
}
