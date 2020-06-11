use crate::{
    domains::Trajectory,
    fa::StateUpdate,
    Function,
    Handler,
    Parameterised,
};

#[derive(Debug, Parameterised)]
pub struct GradientMC<V> {
    #[weights]
    pub v_func: V,

    pub gamma: f64,
}

impl<V> GradientMC<V> {
    pub fn new(v_func: V, gamma: f64) -> Self { GradientMC { v_func, gamma } }
}

impl<'m, S, A, V> Handler<&'m Trajectory<S, A>> for GradientMC<V>
where V: Function<(&'m S,), Output = f64> + Handler<StateUpdate<&'m S, f64>>
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, trajectory: &'m Trajectory<S, A>) -> Result<(), ()> {
        let mut sum = 0.0;

        trajectory.iter().rev().for_each(|transition| {
            sum = transition.reward + self.gamma * sum;

            let from = transition.from.state();
            let pred = self.v_func.evaluate((from,));

            // TODO: Use the result properly.
            self.v_func
                .handle(StateUpdate {
                    state: from,
                    error: sum - pred,
                })
                .ok();
        });

        Ok(())
    }
}
