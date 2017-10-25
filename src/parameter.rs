use std::cmp::max;
use std::f64;
use std::ops::{Add, Sub, Mul, Div};


#[derive(Clone, Copy)]
pub enum Parameter {
    Fixed(f64),
    Exponential {
        init: f64,
        floor: f64,
        count: u32,
        decay: f64,
    },
    Polynomial {
        init: f64,
        floor: f64,
        count: u32,
        exponent: f64,
    },
}

impl Parameter {
    pub fn fixed(value: f64) -> Parameter {
        Parameter::Fixed(value)
    }

    pub fn exponential(init: f64, floor: f64, decay: f64) -> Parameter {
        Parameter::Exponential {
            init: init,
            floor: floor,
            count: 0,
            decay: decay,
        }
    }

    pub fn polynomial(init: f64, floor: f64, exponent: f64) -> Parameter {
        Parameter::Polynomial {
            init: init,
            floor: floor,
            count: 1,
            exponent: exponent,
        }
    }

    pub fn value(&self) -> f64 {
        match self {
            &Parameter::Fixed(v) => v,

            &Parameter::Exponential { init: i, floor: f, count: c, decay: d } => {
                f64::max(i * d.powf(c as f64), f)
            }

            &Parameter::Polynomial { init: i, floor: f, count: c, exponent: e } => {
                f64::max(i / (c as f64).powf(e), f)
            }
        }
    }

    pub fn to_fixed(self) -> Parameter {
        Parameter::Fixed(self.value())
    }

    pub fn step(self) -> Parameter {
        match self {
            Parameter::Fixed(_) => self,
            Parameter::Exponential { init: i, floor: f, count: c, decay: d } => {
                Parameter::Exponential {
                    init: i,
                    floor: f,
                    count: c.saturating_add(1),
                    decay: d,
                }
            }
            Parameter::Polynomial { init: i, floor: f, count: c, exponent: e } => {
                Parameter::Polynomial {
                    init: i,
                    floor: f,
                    count: c.saturating_add(1),
                    exponent: e,
                }
            }
        }
    }

    pub fn back(self) -> Parameter {
        match self {
            Parameter::Fixed(_) => self,
            Parameter::Exponential { init: i, floor: f, count: c, decay: d } => {
                Parameter::Exponential {
                    init: i,
                    floor: f,
                    count: c.saturating_sub(1),
                    decay: d,
                }
            }
            Parameter::Polynomial { init: i, floor: f, count: c, exponent: e } => {
                Parameter::Polynomial {
                    init: i,
                    floor: f,
                    count: max(1, c.saturating_sub(1)),
                    exponent: e,
                }
            }
        }
    }
}

impl Into<Parameter> for f64 {
    fn into(self) -> Parameter {
        Parameter::Fixed(self)
    }
}


macro_rules! impl_op {
    ($name: ident, $num_type: ty, $fn_name: ident, $op: tt) => {
        impl $name<$num_type> for Parameter {
            type Output = $num_type;

            fn $fn_name(self, other: $num_type) -> $num_type {
                self.value() $op other
            }
        }

        impl $name<Parameter> for $num_type {
            type Output = $num_type;

            fn $fn_name(self, other: Parameter) -> $num_type {
                self $op other.value()
            }
        }

        impl $name<Parameter> for Parameter {
            type Output = $num_type;

            fn $fn_name(self, other: Parameter) -> $num_type {
                self.value() $op other.value()
            }
        }
    }
}

impl_op!(Add, f64, add, +);
impl_op!(Sub, f64, sub, -);
impl_op!(Mul, f64, mul, *);
impl_op!(Div, f64, div, /);


#[cfg(test)]
mod tests {
    use super::Parameter;

    #[test]
    fn test_fixed() {
        let mut p = Parameter::fixed(1.0);

        assert_eq!(p.value(), 1.0);

        for _ in 0..1000 {
            p = p.step();
            assert_eq!(p.value(), 1.0);
        }

        for _ in 0..1000 {
            p = p.back();
            assert_eq!(p.value(), 1.0);
        }
    }

    #[test]
    fn test_exponential() {
        let mut p = Parameter::exponential(1.0, 0.5, 0.9);

        assert!((p.value() - 1.0).abs() < 1e-7);

        p = p.step();
        assert!((p.value() - 0.9).abs() < 1e-7);

        p = p.step();
        assert!((p.value() - 0.81).abs() < 1e-7);

        p = p.step();
        assert!((p.value() - 0.729).abs() < 1e-7);

        p = p.back();
        assert!((p.value() - 0.81).abs() < 1e-7);

        p = p.back();
        assert!((p.value() - 0.9).abs() < 1e-7);

        p = p.back();
        assert!((p.value() - 1.0).abs() < 1e-7);

        p = p.back();
        assert!((p.value() - 1.0).abs() < 1e-7);

        for _ in 0..1000 {
            p = p.step();
            assert!(p.value() >= 0.5);
        }

        assert!((p.value() - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_polynomial() {
        let mut p = Parameter::polynomial(1.0, 0.1, 0.6);

        assert!((p.value() - 1.0).abs() < 1e-7);

        p = p.step();
        assert!((p.value() - 0.659753955386447).abs() < 1e-7);

        p = p.step();
        assert!((p.value() - 0.517281857971786).abs() < 1e-7);

        p = p.step();
        assert!((p.value() - 0.435275281648062).abs() < 1e-7);

        p = p.back();
        assert!((p.value() - 0.517281857971786).abs() < 1e-7);

        p = p.back();
        assert!((p.value() - 0.659753955386447).abs() < 1e-7);

        p = p.back();
        assert!((p.value() - 1.0).abs() < 1e-7);

        p = p.back();
        assert!((p.value() - 1.0).abs() < 1e-7);

        for _ in 0..1000 {
            p = p.step();
            assert!(p.value() >= 0.1);
        }

        assert!((p.value() - 0.1).abs() < 1e-7);
    }

    #[test]
    fn test_to_fixed() {
        let mut p = Parameter::exponential(1.0, 0.5, 0.9);
        p = p.step().to_fixed();

        assert!((p.value() - 0.9).abs() < 1e-7);

        p = p.step().step().step().back().back();
        assert!((p.value() - 0.9).abs() < 1e-7);
    }
}
