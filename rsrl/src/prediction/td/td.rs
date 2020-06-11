use crate::{
    domains::{Observation, Transition},
    fa::StateUpdate,
    Function,
    Handler,
};

#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Response<R> {
    pub td_error: f64,
    pub vfunc_response: R,
}

#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct TD<V> {
    #[weights]
    pub v_func: V,

    pub gamma: f64,
}

impl<V> TD<V> {
    pub fn new(v_func: V, gamma: f64) -> Self { TD { v_func, gamma } }
}

impl<'m, S, A, V> Handler<&'m Transition<S, A>> for TD<V>
where V: Function<(&'m S,), Output = f64> + Handler<StateUpdate<&'m S, f64>>
{
    type Response = Response<V::Response>;
    type Error = V::Error;

    fn handle(&mut self, transition: &'m Transition<S, A>) -> Result<Self::Response, Self::Error> {
        let from = transition.from.state();
        let pred = self.v_func.evaluate((from,));

        let td_error = match transition.to {
            Observation::Terminal(_) => transition.reward - pred,
            Observation::Full(ref to) | Observation::Partial(ref to) => {
                transition.reward + self.gamma * self.v_func.evaluate((to,)) - pred
            },
        };

        self.v_func
            .handle(StateUpdate {
                state: from,
                error: td_error,
            })
            .map(|r| Response {
                td_error,
                vfunc_response: r,
            })
    }
}
