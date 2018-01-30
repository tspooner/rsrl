//! Agent memory representation module.

use {Parameter, Vector};

pub enum Trace {
    Accumulating {
        lambda: Parameter,
        eligibility: Vector,
    },
    Replacing {
        lambda: Parameter,
        eligibility: Vector,
    },
    Null {
        eligibility: Vector
    }
    // TODO: Dutch traces (need to be able to share alpha parameter)
}

impl Trace {
    pub fn accumulating<T: Into<Parameter>>(lambda: T, activation: usize) -> Trace {
        Trace::Accumulating {
            lambda: lambda.into(),
            eligibility: Vector::zeros((activation,)),
        }
    }

    pub fn replacing<T: Into<Parameter>>(lambda: T, activation: usize) -> Trace {
        Trace::Replacing {
            lambda: lambda.into(),
            eligibility: Vector::zeros((activation,)),
        }
    }

    pub fn null(activation: usize) -> Trace {
        Trace::Null {
            eligibility: Vector::zeros((activation,)),
        }
    }

    pub fn get(&self) -> Vector {
        match self {
            &Trace::Accumulating { ref eligibility, .. } |
            &Trace::Replacing { ref eligibility, .. } |
            &Trace::Null { ref eligibility } => eligibility.clone(),
        }
    }

    pub fn decay(&mut self, rate: f64) {
        match self {
            &mut Trace::Accumulating { ref mut eligibility, lambda } |
            &mut Trace::Replacing { ref mut eligibility, lambda } => {
                *eligibility *= rate*lambda;
            },
            &mut Trace::Null { ref mut eligibility } => *eligibility *= rate,
        }
    }

    pub fn update(&mut self, phi: &Vector) {
        match self {
            &mut Trace::Accumulating { ref mut eligibility, .. } => {
                *eligibility += phi;
            }
            &mut Trace::Replacing { ref mut eligibility, .. } => {
                eligibility.zip_mut_with(phi, |val, &p| {
                    *val = f64::max(-1.0, f64::min(1.0, *val + p));
                });
            },
            &mut Trace::Null { ref mut eligibility } => *eligibility = phi.to_owned(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::Trace;
    use ndarray::{ArrayBase, arr1};

    #[test]
    fn test_accumulating() {
        let mut trace = Trace::Accumulating {
            lambda: 0.95.into(),
            eligibility: arr1(&[0.0f64; 10]),
        };

        assert_eq!(trace.get(), arr1(&[0.0f64; 10]));

        trace.decay(1.0);
        assert_eq!(trace.get(), arr1(&[0.0f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), arr1(&[1.0f64; 10]));

        trace.decay(1.0);
        assert_eq!(trace.get(), arr1(&[0.95f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), arr1(&[1.95f64; 10]));
    }

    #[test]
    fn test_replacing() {
        let mut trace = Trace::Replacing {
            lambda: 0.95.into(),
            eligibility: arr1(&[0.0f64; 10]),
        };

        assert_eq!(trace.get(), arr1(&[0.0f64; 10]));

        trace.decay(1.0);
        assert_eq!(trace.get(), arr1(&[0.0f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), arr1(&[1.0f64; 10]));

        trace.decay(1.0);
        assert_eq!(trace.get(), arr1(&[0.95f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), arr1(&[1.0f64; 10]));
    }
}
