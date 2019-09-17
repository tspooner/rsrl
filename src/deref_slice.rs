use ndarray::Array1;

pub trait DerefSlice {
    fn deref_slice(&self) -> &[f64];
}

impl DerefSlice for [f64] {
    fn deref_slice(&self) -> &[f64] { self }
}

impl DerefSlice for Vec<f64> {
    fn deref_slice(&self) -> &[f64] { self }
}

impl DerefSlice for Array1<f64> {
    fn deref_slice(&self) -> &[f64] {
        unsafe { ::std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}
