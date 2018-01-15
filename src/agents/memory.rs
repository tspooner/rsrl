//! Agent memory representation module.

use Parameter;
use ndarray::Array1;


pub enum TraceType {
    Accumulating,
    Replacing,
}


pub struct Trace {
    pub trace_type: TraceType,

    pub lambda: Parameter,
    pub eligibility: Array1<f64>,
}

impl Trace {
    pub fn new<T: Into<Parameter>>(trace_type: TraceType, lambda: T, activation: usize) -> Trace {
        Trace {
            trace_type: trace_type,

            lambda: lambda.into(),
            eligibility: Array1::zeros((activation,)),
        }
    }

    pub fn accumulating<T: Into<Parameter>>(lambda: T, activation: usize) -> Trace {
        Trace::new(TraceType::Accumulating, lambda, activation)
    }

    pub fn replacing<T: Into<Parameter>>(lambda: T, activation: usize) -> Trace {
        Trace::new(TraceType::Replacing, lambda, activation)
    }

    pub fn get(&self) -> Array1<f64> {
        self.eligibility.clone()
    }

    pub fn decay(&mut self, rate: f64) {
        self.eligibility *= rate;
    }

    pub fn update(&mut self, activation: &Array1<f64>) {
        match self.trace_type {
            TraceType::Accumulating => self.eligibility += activation,
            TraceType::Replacing => {
                self.eligibility.zip_mut_with(activation, |val, &a| {
                    *val = f64::max(-1.0, f64::min(1.0, *val + a));
                });
            },
        }
    }
}


#[cfg(test)]
mod tests {
    use super::Trace;
    use ndarray::{ArrayBase, arr1};

    #[test]
    fn test_accumulating() {
        let mut trace = Trace::accumulating(0.95, 10);

        assert_eq!(trace.get(), arr1(&[0.0f64; 10]));

        let l = trace.lambda.value();
        trace.decay(l);
        assert_eq!(trace.get(), arr1(&[0.0f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), arr1(&[1.0f64; 10]));

        let l = trace.lambda.value();
        trace.decay(l);
        assert_eq!(trace.get(), arr1(&[0.95f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), arr1(&[1.95f64; 10]));
    }

    #[test]
    fn test_replacing() {
        let mut trace = Trace::replacing(0.95, 10);

        assert_eq!(trace.get(), arr1(&[0.0f64; 10]));

        let l = trace.lambda.value();
        trace.decay(l);
        assert_eq!(trace.get(), arr1(&[0.0f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), arr1(&[1.0f64; 10]));

        let l = trace.lambda.value();
        trace.decay(l);
        assert_eq!(trace.get(), arr1(&[0.95f64; 10]));

        trace.update(&arr1(&[1.0f64; 10]));
        assert_eq!(trace.get(), arr1(&[1.0f64; 10]));
    }
}
