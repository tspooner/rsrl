pub struct Partitions {
    pub lb: f64,
    pub ub: f64,

    pub density: usize,
}

impl Partitions {
    pub fn new(lb: f64, ub: f64, density: usize) -> Partitions {
        Partitions {
            lb: lb,
            ub: ub,
            density: density,
        }
    }

    #[inline]
    pub fn dimensionality(partitions: &[Partitions]) -> usize {
        partitions.iter().fold(1, |acc, p| acc * p.density)
    }

    pub fn to_partition(&self, val: f64) -> usize {
        let clipped = clip!(self.lb, val, self.ub);

        let diff = clipped - self.lb;
        let range = self.ub - self.lb;

        let i = ((self.density as f64) * diff / range).floor() as usize;

        if i == self.density { i - 1 } else { i }
    }

    pub fn centres(&self) -> Vec<f64> {
        let w = self.partition_width();
        let hw = w / 2.0;

        (0..self.density).map(|i| self.lb + w*(i as f64) - hw).collect()
    }

    pub fn partition_width(&self) -> f64 {
        (self.ub - self.lb) / self.density as f64
    }
}
