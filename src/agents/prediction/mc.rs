use Parameter;
use agents::PredictionAgent;
use fa::VFunction;
use geometry::Space;
use std::marker::PhantomData;


pub struct EveryVisitMC<S: Space, V: VFunction<S>> {
    pub v_func: V,
    observations: Vec<(S::Repr, f64)>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, V> EveryVisitMC<S, V>
    where V: VFunction<S>
{
    pub fn new<T1, T2>(v_func: V, alpha: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        EveryVisitMC {
            v_func: v_func,
            observations: vec![],

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }

    pub fn propagate(&mut self) {
        let mut sum = 0.0;

        for (s, r) in self.observations.drain(0..).rev() {
            sum = r + self.gamma*sum;

            let v_est = self.v_func.evaluate(&s);
            self.v_func.update(&s, self.alpha*(sum - v_est));
        }
    }
}

impl<S: Space, V> PredictionAgent<S> for EveryVisitMC<S, V>
    where V: VFunction<S>
{
    fn evaluate(&self, s: &S::Repr) -> f64 {
        self.v_func.evaluate(s)
    }

    fn handle_transition(&mut self, s: &S::Repr, _: &S::Repr, r: f64) -> Option<f64> {
        self.observations.push((s.clone(), r));

        None
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.propagate();

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}
