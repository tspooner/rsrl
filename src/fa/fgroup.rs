use super::{Function, Parameterised, VFunction, QFunction};
use geometry::Space;
use std::marker::PhantomData;


#[derive(Clone)]
pub struct VFunctionGroup<S, V: VFunction<S>>(Vec<V>, PhantomData<S>)
    where S: Space;

impl<S, V: VFunction<S>> VFunctionGroup<S, V>
    where S: Space
{
    pub fn new(functions: Vec<V>) -> Self {
        VFunctionGroup(functions, PhantomData)
    }
}

impl<S, V: VFunction<S>> Function<S::Repr, Vec<f64>> for VFunctionGroup<S, V>
    where S: Space
{
    fn evaluate(&self, state: &S::Repr) -> Vec<f64> {
        self.0.iter().map(|f| f.evaluate(state)).collect()
    }
}

impl<S, V: VFunction<S>> Parameterised<S::Repr, Vec<f64>> for VFunctionGroup<S, V>
    where S: Space
{
    fn update(&mut self, state: &S::Repr, mut errors: Vec<f64>) {
        let mut index = 0;

        for e in errors.drain(..) {
            self.0[index].update(state, e);
            index += 1;
        }
    }

    fn equivalent(&self, other: &Self) -> bool {
        !self.0.iter().zip(other.0.iter())
            .any(|(ref f1, ref f2)| !f1.equivalent(f2))
    }
}

impl<S, V: VFunction<S>> QFunction<S> for VFunctionGroup<S, V>
    where S: Space
{
    fn evaluate_action(&self, state: &S::Repr, action: usize) -> f64 {
        self.0[action].evaluate(state)
    }

    fn update_action(&mut self, state: &S::Repr, action: usize, error: f64) {
        self.0[action].update(state, error);
    }
}


#[cfg(test)]
mod tests {
    use super::{VFunctionGroup, Function, Parameterised, VFunction, QFunction};
    use geometry::RegularSpace;
    use geometry::dimensions::Continuous;


    pub struct Mock(f64);

    impl Function<Vec<f64>, f64> for Mock {
        fn evaluate(&self, _: &Vec<f64>) -> f64 { self.0 }
    }

    impl Parameterised<Vec<f64>, f64> for Mock {
        fn update(&mut self, _: &Vec<f64>, e: f64) { self.0 = e; }
        fn equivalent(&self, other: &Self) -> bool {
            (self.0 - other.0).abs() < 1e-5
        }
    }

    impl VFunction<RegularSpace<Continuous>> for Mock {}


    #[test]
    fn test_function() {
        let fg = VFunctionGroup::new(vec![Mock(0.0), Mock(1.0), Mock(2.0)]);

        assert_eq!(fg.evaluate(&vec![]), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_parameterised() {
        let mut fg = VFunctionGroup::new(vec![Mock(0.0), Mock(1.0), Mock(2.0)]);

        fg.update(&vec![], vec![4.0, -1.0, 8.0]);

        assert_eq!(fg.evaluate(&vec![]), vec![4.0, -1.0, 8.0]);
    }

    #[test]
    fn test_equivalency() {
        let fg1 = VFunctionGroup::new(vec![Mock(0.0), Mock(1.0), Mock(2.0)]);
        let fg2 = VFunctionGroup::new(vec![Mock(2.0), Mock(1.0), Mock(2.0)]);
        let fg3 = VFunctionGroup::new(vec![Mock(0.0), Mock(1.0), Mock(2.0)]);

        assert!(fg1.equivalent(&fg1));
        assert!(fg1.equivalent(&fg3));
        assert!(fg2.equivalent(&fg2));
        assert!(fg3.equivalent(&fg1));
        assert!(fg3.equivalent(&fg3));

        assert!(!fg1.equivalent(&fg2));
        assert!(!fg2.equivalent(&fg1));
        assert!(!fg2.equivalent(&fg3));
        assert!(!fg3.equivalent(&fg2));
    }

    #[test]
    fn test_qfunction() {
        let fg = VFunctionGroup::new(vec![Mock(0.0), Mock(1.0), Mock(2.0)]);

        assert_eq!(fg.evaluate_action(&vec![], 0), 0.0);
        assert_eq!(fg.evaluate_action(&vec![], 1), 1.0);
        assert_eq!(fg.evaluate_action(&vec![], 2), 2.0);
    }

    #[test]
    fn test_boxed_vfuncs() {
        let mut fg = VFunctionGroup::new(vec![
            Box::new(Mock(0.0)),
            Box::new(Mock(1.0)),
            Box::new(Mock(2.0))
        ]);

        assert_eq!(fg.evaluate(&vec![]), vec![0.0, 1.0, 2.0]);

        assert_eq!(fg.evaluate_action(&vec![], 0), 0.0);
        assert_eq!(fg.evaluate_action(&vec![], 1), 1.0);
        assert_eq!(fg.evaluate_action(&vec![], 2), 2.0);

        fg.update(&vec![], vec![4.0, -1.0, 8.0]);

        assert_eq!(fg.evaluate(&vec![]), vec![4.0, -1.0, 8.0]);
    }
}
