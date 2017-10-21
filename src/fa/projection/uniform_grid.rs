use super::Projection;
use ndarray::Array1;
use geometry::{Space, RegularSpace};
use geometry::dimensions::{Dimension, Continuous, Partitioned};


pub struct UniformGrid {
    n_features: usize,
    input_space: RegularSpace<Partitioned>,
}

impl UniformGrid {
    pub fn new(input_space: RegularSpace<Partitioned>) -> Self {
        let n_features = input_space.span().into();

        UniformGrid {
            n_features: n_features,
            input_space: input_space,
        }
    }

    fn hash(&self, input: &[f64]) -> usize {
        let mut in_it = input.iter().rev();
        let mut d_it = self.input_space.iter().rev();

        let acc = d_it.next().unwrap().convert(*in_it.next().unwrap());

        d_it.zip(in_it)
            .fold(acc, |acc, (d, v)| d.convert(*v) + d.density() * acc)
    }
}

impl Projection<RegularSpace<Continuous>> for UniformGrid {
    fn project(&self, input: &Vec<f64>) -> Array1<f64> {
        let mut p = Array1::<f64>::zeros(self.n_features);
        p[self.hash(input)] = 1.0;

        p
    }

    fn dim(&self) -> usize {
        self.n_features
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.dim() == other.dim()
    }
}


#[cfg(test)]
mod tests {
    use super::{Projection, UniformGrid};
    use geometry::RegularSpace;
    use geometry::dimensions::Partitioned;


    #[test]
    fn test_1d() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));

        let mut t = UniformGrid::new(ds);

        assert_eq!(t.dim(), 10);

        for i in 0..10 {
            let out = t.project(&vec![i as u32 as f64]).to_vec();

            let mut expected = vec![0.0; 10];
            expected[i] = 1.0;

            assert_eq!(out, expected);
        }
    }

    #[test]
    fn test_2d() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));

        let mut t = UniformGrid::new(ds);

        assert_eq!(t.dim(), 100);

        for i in 0..10 {
            for j in 0..10 {
                let out =
                    t.project(&vec![i as u32 as f64, j as u32 as f64]).to_vec();

                let mut expected = vec![0.0; 100];
                expected[j*10 + i] = 1.0;

                assert_eq!(out, expected);
            }
        }
    }

    #[test]
    fn test_3d() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));

        let mut t = UniformGrid::new(ds);

        assert_eq!(t.dim(), 1000);

        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let out = t.project(&vec![i as u32 as f64,
                                              j as u32 as f64,
                                              k as u32 as f64]).to_vec();

                    let mut expected = vec![0.0; 1000];
                    expected[k*100 + j*10 + i] = 1.0;

                    assert_eq!(out, expected);
                }
            }
        }
    }
}
