use crate::geometry::Vector;

pub trait DerefSlice {
    fn deref_slice(&self) -> &[f64];
}

impl DerefSlice for [f64] {
    fn deref_slice(&self) -> &[f64] { self }
}

impl DerefSlice for Vec<f64> {
    fn deref_slice(&self) -> &[f64] { self }
}

impl DerefSlice for Vector<f64> {
    fn deref_slice(&self) -> &[f64] {
        unsafe { ::std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}
