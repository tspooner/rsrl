use crate::geometry::Vector;
use super::Transform;

// f(x) ≜ x
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Identity;

macro_rules! impl_identity {
    ($type:ty; $grad:expr) => {
        impl Transform<$type> for Identity {
            type Output = $type;

            fn transform(&self, x: $type) -> $type { x }

            fn grad(&self, _: $type) -> $type { $grad }

            fn grad_scaled(&self, _: $type, error: $type) -> $type { error }
        }
    }
}

impl_identity!(f64; 1.0);
impl_identity!([f64; 2]; [1.0; 2]);
impl_identity!([f64; 3]; [1.0; 3]);

impl Transform<Vec<f64>> for Identity {
    type Output = Vec<f64>;

    fn transform(&self, x: Vec<f64>) -> Vec<f64> { x }

    fn grad(&self, x: Vec<f64>) -> Vec<f64> { x.into_iter().map(|_| 1.0).collect() }

    fn grad_scaled(&self, _: Vec<f64>, errors: Vec<f64>) -> Vec<f64> { errors }
}

impl Transform<Vector<f64>> for Identity {
    type Output = Vector<f64>;

    fn transform(&self, x: Vector<f64>) -> Vector<f64> { x }

    fn grad(&self, mut x: Vector<f64>) -> Vector<f64> { x.fill(1.0); x }

    fn grad_scaled(&self, _: Vector<f64>, errors: Vector<f64>) -> Vector<f64> { errors }
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use super::{Identity, Transform, Vector};

    #[test]
    fn test_scalar() {
        fn prop_transform(val: f64) -> bool {
            (Identity.transform(val) - val).abs() < 1e-7
        }

        fn prop_grad(val: f64) -> bool {
            (Identity.grad(val) - 1.0).abs() < 1e-7
        }

        quickcheck(prop_transform as fn(f64) -> bool);
        quickcheck(prop_grad as fn(f64) -> bool);
    }

    #[test]
    fn test_pair() {
        fn prop_transform(val: (f64, f64)) -> bool {
            let t = Identity.transform([val.0, val.1]);

            (t[0] - val.0).abs() < 1e-7 && (t[1] - val.1).abs() < 1e-7
        }

        fn prop_grad(val: (f64, f64)) -> bool {
            let g = Identity.grad([val.0, val.1]);

            (g[0] - 1.0).abs() < 1e-7 && (g[1] - 1.0).abs() < 1e-7
        }

        quickcheck(prop_transform as fn((f64, f64)) -> bool);
        quickcheck(prop_grad as fn((f64, f64)) -> bool);
    }

    #[test]
    fn test_triple() {
        fn prop_transform(val: (f64, f64, f64)) -> bool {
            let t = Identity.transform([val.0, val.1, val.2]);

            (t[0] - val.0).abs() < 1e-7 && (t[1] - val.1).abs() < 1e-7 && (t[2] - val.2).abs() < 1e-7
        }

        fn prop_grad(val: (f64, f64, f64)) -> bool {
            let g = Identity.grad([val.0, val.1, val.2]);

            (g[0] - 1.0).abs() < 1e-7 && (g[1] - 1.0).abs() < 1e-7 && (g[2] - 1.0).abs() < 1e-7
        }

        quickcheck(prop_transform as fn((f64, f64, f64)) -> bool);
        quickcheck(prop_grad as fn((f64, f64, f64)) -> bool);
    }

    #[test]
    fn test_vector() {
        fn prop_transform(val: Vec<f64>) -> bool {
            let t = Identity.transform(Vector::from_vec(val.clone()));

            t.into_iter().zip(val.into_iter()).all(|(v1, v2)| (v1 - v2).abs() < 1e-7)
        }

        fn prop_grad(val: Vec<f64>) -> bool {
            let g = Identity.grad(Vector::from_vec(val));

            g.into_iter().all(|v| (v - 1.0).abs() < 1e-7)
        }

        quickcheck(prop_transform as fn(Vec<f64>) -> bool);
        quickcheck(prop_grad as fn(Vec<f64>) -> bool);
    }
}
