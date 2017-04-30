use super::{Function, Parameterised, VFunction, QFunction};
use geometry::Space;
use std::marker::PhantomData;


pub struct VFunctionGroup<S: Space, V: VFunction<S>>(Vec<V>, PhantomData<S>);

impl<S: Space, V: VFunction<S>> VFunctionGroup<S, V>
{
    pub fn new(functions: Vec<V>) -> Self {
        VFunctionGroup(functions, PhantomData)
    }
}

impl<S: Space, V: VFunction<S>> Function<S::Repr, Vec<f64>> for VFunctionGroup<S, V>
{
    fn evaluate(&self, state: &S::Repr) -> Vec<f64> {
        self.0.iter().map(|f| f.evaluate(state)).collect()
    }
}

impl<S: Space, V: VFunction<S>> Parameterised<S::Repr, Vec<f64>> for VFunctionGroup<S, V>
{
    fn update(&mut self, state: &S::Repr, mut errors: Vec<f64>) {
        let mut index = 0;

        for e in errors.drain(..) {
            self.0[index].update(state, e);
            index += 1;
        }
    }
}

impl<S: Space, V: VFunction<S>> QFunction<S> for VFunctionGroup<S, V>
{
    fn evaluate_action(&self, state: &S::Repr, action: usize) -> f64 {
        self.0[action].evaluate(state)
    }

    fn update_action(&mut self, state: &S::Repr, action: usize, error: f64) {
        self.0[action].update(state, error);
    }
}
