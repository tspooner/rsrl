use Parameter;
use agents::{Agent, LinearAgent, Predictor, TDPredictor};
use fa::{Function, VFunction, Projector, Projection, Linear};
use geometry::Space;
use ndarray::Array2;


/// Gradient temporal difference learning, version 2.
pub struct GTD2<S: Space, P: Projector<S>> {
    pub fa_theta: Linear<S, P>,
    pub fa_w: Linear<S, P>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S: Space, P: Projector<S>> GTD2<S, P> {
    pub fn new<T1, T2, T3>(fa_theta: Linear<S, P>,
                           fa_w: Linear<S, P>,
                           alpha: T1,
                           beta: T2,
                           gamma: T3)
                           -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>
    {
        if !fa_theta.projector.equivalent(&fa_w.projector) {
            panic!("fa_theta and fa_w must be equivalent function approximators.")
        }

        GTD2 {
            fa_theta: fa_theta,
            fa_w: fa_w,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }

    fn compute_error_with_phi(&self, phi_s: &Projection, phi_ns: &Projection, reward: f64) -> f64 {
        let v: f64 = self.fa_theta.evaluate_phi(phi_s);
        let nv: f64 = self.fa_theta.evaluate_phi(phi_ns);

        reward + self.gamma*nv - v
    }

    fn handle_error_with_phi(&mut self, phi_s: Projection, phi_ns: Projection, td_error: f64) {
        let td_estimate = self.fa_w.evaluate_phi(&phi_s);

        self.fa_w.update_phi(&phi_s, self.beta*(td_error - td_estimate));

        {
            let phi_s = self.fa_theta.projector.expand_projection(phi_s);
            let phi_ns = self.fa_theta.projector.expand_projection(phi_ns);
            let update = &phi_s - &(self.gamma.value()*&phi_ns);

            self.fa_theta.update_phi(&Projection::Dense(update), self.alpha*td_estimate);
        }
    }
}

impl<S: Space, P: Projector<S>> Agent<S> for GTD2<S, P> {
    type Sample = (S::Repr, S::Repr, f64);

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let phi_s = self.fa_theta.projector.project(&sample.0);
        let phi_ns = self.fa_theta.projector.project(&sample.1);

        let td_error = self.compute_error_with_phi(&phi_s, &phi_ns, sample.2);

        self.handle_error_with_phi(phi_s, phi_ns, td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S: Space, P: Projector<S>> LinearAgent<S> for GTD2<S, P> {
    fn weights(&self) -> Array2<f64> {
        self.fa_theta.weights.clone()
    }
}

impl<S: Space, P: Projector<S>> Predictor<S> for GTD2<S, P> {
    fn evaluate(&self, s: &S::Repr) -> f64 {
        self.fa_theta.evaluate(s)
    }
}

impl<S: Space, P: Projector<S>> TDPredictor<S> for GTD2<S, P> {
    fn handle_error(&mut self, sample: &Self::Sample, td_error: f64) {
        let phi_s = self.fa_theta.projector.project(&sample.0);
        let phi_ns = self.fa_theta.projector.project(&sample.1);

        self.handle_error_with_phi(phi_s, phi_ns, td_error);
    }

    fn compute_error(&self, sample: &Self::Sample) -> f64 {
        let phi_s = self.fa_theta.projector.project(&sample.0);
        let phi_ns = self.fa_theta.projector.project(&sample.1);

        self.compute_error_with_phi(&phi_s, &phi_ns, sample.2)
    }
}


/// Temporal difference learning with corrections.
///
/// Note: `beta` should be greater than `alpha`.
pub struct TDC<S: Space, P: Projector<S>> {
    pub fa_theta: Linear<S, P>,
    pub fa_w: Linear<S, P>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S: Space, P: Projector<S>> TDC<S, P> {
    pub fn new<T1, T2, T3>(fa_theta: Linear<S, P>,
                           fa_w: Linear<S, P>,
                           alpha: T1,
                           beta: T2,
                           gamma: T3)
                           -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>
    {
        if !fa_theta.projector.equivalent(&fa_w.projector) {
            panic!("fa_theta and fa_w must be equivalent function approximators.")
        }

        TDC {
            fa_theta: fa_theta,
            fa_w: fa_w,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }

    fn compute_error_with_phi(&self, phi_s: &Projection, phi_ns: &Projection, reward: f64) -> f64 {
        let v: f64 = self.fa_theta.evaluate_phi(phi_s);
        let nv: f64 = self.fa_theta.evaluate_phi(phi_ns);

        reward + self.gamma*nv - v
    }

    fn handle_error_with_phi(&mut self, phi_s: Projection, phi_ns: Projection, td_error: f64) {
        let td_estimate = self.fa_w.evaluate_phi(&phi_s);

        self.fa_w.update_phi(&phi_s, self.beta*(td_error - td_estimate));

        {
            let phi_s = self.fa_theta.projector.expand_projection(phi_s);
            let phi_ns = self.fa_theta.projector.expand_projection(phi_ns);
            let update = &(td_error*&phi_s) - &(self.gamma.value()*td_estimate*&phi_ns);

            self.fa_theta.update_phi(&Projection::Dense(update), self.alpha.value());
        }
    }
}

impl<S: Space, P: Projector<S>> Agent<S> for TDC<S, P> {
    type Sample = (S::Repr, S::Repr, f64);

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let phi_s = self.fa_theta.projector.project(&sample.0);
        let phi_ns = self.fa_theta.projector.project(&sample.1);

        let td_error = self.compute_error_with_phi(&phi_s, &phi_ns, sample.2);

        self.handle_error_with_phi(phi_s, phi_ns, td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S: Space, P: Projector<S>> LinearAgent<S> for TDC<S, P> {
    fn weights(&self) -> Array2<f64> {
        self.fa_theta.weights.clone()
    }
}

impl<S: Space, P: Projector<S>> Predictor<S> for TDC<S, P> {
    fn evaluate(&self, s: &S::Repr) -> f64 {
        self.fa_theta.evaluate(s)
    }
}

impl<S: Space, P: Projector<S>> TDPredictor<S> for TDC<S, P> {
    fn handle_error(&mut self, sample: &Self::Sample, td_error: f64) {
        let phi_s = self.fa_theta.projector.project(&sample.0);
        let phi_ns = self.fa_theta.projector.project(&sample.1);

        self.handle_error_with_phi(phi_s, phi_ns, td_error);
    }

    fn compute_error(&self, sample: &Self::Sample) -> f64 {
        let phi_s = self.fa_theta.projector.project(&sample.0);
        let phi_ns = self.fa_theta.projector.project(&sample.1);

        self.compute_error_with_phi(&phi_s, &phi_ns, sample.2)
    }
}


// TODO:
// ABQ(lambda) - https://arxiv.org/pdf/1702.03006.pdf
// GQ(lambda) - http://agi-conf.org/2010/wp-content/uploads/2009/06/paper_21.pdf
// GTD(lambda) - https://era.library.ualberta.ca/files/8s45q967t/Hamid_Maei_PhDThesis.pdf
// True online GTD(lambda) - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.487.2451&rep=rep1&type=pdf
// GTD2(lambda)-MP - https://arxiv.org/pdf/1602.08771.pdf
// TDC(lambda)-MP - https://arxiv.org/pdf/1602.08771.pdf
