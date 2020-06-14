use crate::{
    domains::Trajectory,
    fa::StateUpdate,
    Function,
    Handler,
    Parameterised,
};

#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Error<E> {
    pub timestep: usize,
    pub error: E,
}

#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct GradientMC<V> {
    #[weights]
    pub v_func: V,

    pub gamma: f64,
}

impl<'m, S, A, V> Handler<&'m Trajectory<S, A>> for GradientMC<V>
where V: Function<(&'m S,), Output = f64> + Handler<StateUpdate<&'m S, f64>>
{
    type Response = Vec<V::Response>;
    type Error = Error<V::Error>;

    fn handle(&mut self, traj: &'m Trajectory<S, A>) -> Result<Self::Response, Self::Error> {
        let n = traj.n_transitions();
        let mut sum = 0.0;

        traj.iter().rev().enumerate().map(|(t, transition)| {
            sum = transition.reward + self.gamma * sum;

            let from = transition.from.state();
            let pred = self.v_func.evaluate((from,));

            self.v_func.handle(StateUpdate {
                state: from,
                error: sum - pred,
            }).map_err(|e| Error {
                timestep: n - t - 1,
                error: e,
            })
        }).collect()
    }
}
