use matrix::Matrix;

#[derive(Debug)]
pub struct Pairs<'a> {
    pub param: &'a mut Matrix,
    pub gradient: &'a Matrix,
}

pub trait Optimizer {
    fn update<'a, 'b, I>(&'a mut self, pairs: I)
    where
        I: Iterator<Item = Pairs<'b>>;
}

#[derive(Debug)]
pub struct Sgd {
    learning_rate: f64,
}
impl Sgd {
    pub fn new(learning_rate: f64) -> Self {
        Sgd { learning_rate }
    }
}
impl Optimizer for Sgd {
    fn update<'a, 'b, I>(&'a mut self, pairs: I)
    where
        I: Iterator<Item = Pairs<'b>>,
    {
        for t in pairs {
            *t.param -= t.gradient.clone() * self.learning_rate;
        }
    }
}

#[derive(Debug)]
pub struct Momentum {
    learning_rate: f64,
    momentum: f64,
    v: Vec<Matrix>,
}
impl Momentum {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Momentum {
            learning_rate,
            momentum,
            v: Vec::new(),
        }
    }
}
impl Optimizer for Momentum {
    fn update<'a, 'b, I>(&'a mut self, pairs: I)
    where
        I: Iterator<Item = Pairs<'b>>,
    {
        for (i, t) in pairs.enumerate() {
            if self.v.len() < i {
                self.v.push(Matrix::new(t.param.rows(), t.param.columns()));
            }
            self.v[i] =
                (self.v[i].clone() * self.momentum).sub(&(t.gradient.clone() * self.learning_rate));
            *t.param = t.param.clone().add(&self.v[i]);
        }
    }
}
impl Default for Momentum {
    fn default() -> Self {
        Self::new(0.01, 0.9)
    }
}

#[derive(Debug)]
pub struct AdaGrad {
    learning_rate: f64,
    h: Vec<Matrix>,
}
impl AdaGrad {
    pub fn new(learning_rate: f64) -> Self {
        AdaGrad {
            learning_rate,
            h: Vec::new(),
        }
    }
}
impl Optimizer for AdaGrad {
    fn update<'a, 'b, I>(&'a mut self, pairs: I)
    where
        I: Iterator<Item = Pairs<'b>>,
    {
        for (i, t) in pairs.enumerate() {
            if self.h.len() < i {
                self.h.push(Matrix::new(t.param.rows(), t.param.columns()));
            }
            self.h[i] += t.gradient.clone() * t.gradient.clone();
            *t.param -= (t.gradient.clone() * self.learning_rate) / (self.h[i].sqrt() + 1e-7);
        }
    }
}
