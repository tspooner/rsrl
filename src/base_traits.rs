pub trait Function<I: ?Sized, O> {
    fn evaluate(&self, input: &I) -> O;
}


// TODO: Implement binary serialization with compression
pub trait Parameterised<I: ?Sized, T: ?Sized> {
    fn update(&mut self, input: &I, errors: &T);

    // fn load
    // fn save
}
