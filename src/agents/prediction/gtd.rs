use Parameter;
use agents::{Agent, Predictor, TDPredictor};
use fa::{Approximator, VFunction, SimpleLinear, Projection, Projector};
use geometry::Space;

// TODO: Implement TDPredictor for all agents here.

pub struct GTD2<S: Space, P: Projector<S::Repr>> {
    pub fa_theta: SimpleLinear<S::Repr, P>,
    pub fa_w: SimpleLinear<S::Repr, P>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S: Space, P: Projector<S::Repr>> GTD2<S, P> {
    pub fn new<T1, T2, T3>(
        fa_theta: SimpleLinear<S::Repr, P>,
        fa_w: SimpleLinear<S::Repr, P>,
        alpha: T1,
        beta: T2,
        gamma: T3,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        if !(fa_theta.projector.span() == fa_w.projector.span()) {
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
}

impl<S: Space, P: Projector<S::Repr>> Agent for GTD2<S, P> {
    type Sample = (S::Repr, S::Repr, f64);

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let phi_s = self.fa_theta.projector.project(&sample.0);
        let phi_ns = self.fa_theta.projector.project(&sample.1);

        let td_error = sample.2 + self.gamma * self.fa_theta.evaluate_phi(&phi_ns)
            - self.fa_theta.evaluate_phi(&phi_s);
        let td_estimate = self.fa_w.evaluate_phi(&phi_s);

        let span = self.fa_theta.projector.span();
        let update = phi_s.clone().expanded(span) - self.gamma.value() * phi_ns.expanded(span);

        self.fa_w.update_phi(&phi_s, self.beta * (td_error - td_estimate));
        self.fa_theta.update_phi(&Projection::Dense(update), self.alpha * td_estimate);
    }

    fn handle_terminal(&mut self, _: &Self::Sample) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S: Space, P: Projector<S::Repr>> Predictor<S> for GTD2<S, P> {
    fn evaluate(&self, s: &S::Repr) -> f64 { self.fa_theta.evaluate(s).unwrap() }
}

pub struct TDC<S: Space, P: Projector<S::Repr>> {
    pub fa_theta: SimpleLinear<S::Repr, P>,
    pub fa_w: SimpleLinear<S::Repr, P>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S: Space, P: Projector<S::Repr>> TDC<S, P> {
    pub fn new<T1, T2, T3>(
        fa_theta: SimpleLinear<S::Repr, P>,
        fa_w: SimpleLinear<S::Repr, P>,
        alpha: T1,
        beta: T2,
        gamma: T3,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        if !(fa_theta.projector.span() == fa_w.projector.span()) {
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
}

impl<S: Space, P: Projector<S::Repr>> Agent for TDC<S, P> {
    type Sample = (S::Repr, S::Repr, f64);

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let phi_s = self.fa_theta.projector.project(&sample.0);
        let phi_ns = self.fa_theta.projector.project(&sample.1);

        let td_error = sample.2 + self.gamma * self.fa_theta.evaluate_phi(&phi_ns)
            - self.fa_theta.evaluate_phi(&phi_s);
        let td_estimate = self.fa_w.evaluate_phi(&phi_s);

        let span = self.fa_theta.projector.span();
        let update = td_error * phi_s.clone().expanded(span)
            - self.gamma.value() * td_estimate * &phi_ns.expanded(span);

        self.fa_w.update_phi(&phi_s, self.beta * (td_error - td_estimate));
        self.fa_theta.update_phi(&Projection::Dense(update), self.alpha.value());
    }

    fn handle_terminal(&mut self, _: &Self::Sample) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S: Space, P: Projector<S::Repr>> Predictor<S> for TDC<S, P> {
    fn evaluate(&self, s: &S::Repr) -> f64 { self.fa_theta.evaluate(s).unwrap() }
}

// TODO:
// ABQ(lambda) - https://arxiv.org/pdf/1702.03006.pdf
// GQ(lambda) - http://agi-conf.org/2010/wp-content/uploads/2009/06/paper_21.pdf
// GTD(lambda) - https://era.library.ualberta.ca/files/8s45q967t/Hamid_Maei_PhDThesis.pdf
// True online GTD(lambda) - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.487.2451&rep=rep1&type=pdf
// GTD2(lambda)-MP - https://arxiv.org/pdf/1602.08771.pdf
// TDC(lambda)-MP - https://arxiv.org/pdf/1602.08771.pdf
