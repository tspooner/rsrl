use Parameter;
use agents::PredictionAgent;
use fa::{VFunction, Projector, Projection, Linear};
use geometry::Space;


pub struct GTD2<S: Space, P: Projector<S>> {
    pub v_func: Linear<S, P>,
    pub a_func: Linear<S, P>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S: Space, P: Projector<S>> GTD2<S, P> {
    pub fn new<T1, T2, T3>(v_func: Linear<S, P>,
                           a_func: Linear<S, P>,
                           alpha: T1,
                           beta: T2,
                           gamma: T3)
                           -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>
    {
        if !v_func.projector.equivalent(&a_func.projector) {
            panic!("v_func and a_func must be equivalent function approximators.")
        }

        GTD2 {
            v_func: v_func,
            a_func: a_func,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<S: Space, V> PredictionAgent<S> for GTD2<S, V>
    where V: VFunction<S> + Projector<S>
{
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64> {
        let phi_s = self.v_func.projector.project(s);
        let phi_ns = self.v_func.projector.project(ns);

        let td_error = r + self.gamma*self.v_func.evaluate_phi(&phi_ns) -
            self.v_func.evaluate_phi(&phi_s);
        let td_estimate = self.a_func.evaluate_phi(&phi_s);

        self.a_func.update_phi(&phi_s, self.beta*(td_error - td_estimate));

        {
            let phi_s = self.v_func.projector.expand_projection(phi_s);
            let phi_ns = self.v_func.projector.expand_projection(phi_ns);
            let update = &phi_s - &(self.gamma.value()*&phi_ns);

            self.v_func.update_phi(&Projection::Dense(update), self.alpha*td_estimate);
        }

        Some(td_error)
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}


pub struct TDC<S: Space, P: Projector<S>> {
    pub v_func: Linear<S, P>,
    pub a_func: Linear<S, P>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S: Space, P: Projector<S>> TDC<S, P> {
    pub fn new<T1, T2, T3>(v_func: Linear<S, P>,
                           a_func: Linear<S, P>,
                           alpha: T1,
                           beta: T2,
                           gamma: T3)
                           -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>
    {
        if !v_func.projector.equivalent(&a_func.projector) {
            panic!("v_func and a_func must be equivalent function approximators.")
        }

        TDC {
            v_func: v_func,
            a_func: a_func,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<S: Space, V> PredictionAgent<S> for TDC<S, V>
    where V: VFunction<S> + Projector<S>
{
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64> {
        let phi_s = self.v_func.projector.project(s);
        let phi_ns = self.v_func.projector.project(ns);

        let td_error = r + self.gamma*self.v_func.evaluate_phi(&phi_ns) -
            self.v_func.evaluate_phi(&phi_s);
        let td_estimate = self.a_func.evaluate_phi(&phi_s);

        self.a_func.update_phi(&phi_s, self.beta*(td_error - td_estimate));

        {
            let phi_s = self.v_func.projector.expand_projection(phi_s);
            let phi_ns = self.v_func.projector.expand_projection(phi_ns);
            let update = &(td_error*&phi_s) - &(self.gamma.value()*td_estimate*&phi_ns);

            self.v_func.update_phi(&Projection::Dense(update), self.alpha.value());
        }

        Some(td_error)
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}


// TODO:
// GQ(lambda) - http://agi-conf.org/2010/wp-content/uploads/2009/06/paper_21.pdf
// GTD(lambda) - https://era.library.ualberta.ca/files/8s45q967t/Hamid_Maei_PhDThesis.pdf
// True online GTD(lambda) - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.487.2451&rep=rep1&type=pdf
// GTD2(lambda)-MP - https://arxiv.org/pdf/1602.08771.pdf
// TDC(lambda)-MP - https://arxiv.org/pdf/1602.08771.pdf
