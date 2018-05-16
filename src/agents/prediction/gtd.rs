use agents::{Predictor};
use domains::Transition;
use fa::{Approximator, Projection, Projector, SimpleLinear, VFunction};
use {Handler, Parameter};

// TODO: Implement TDPredictor for all agents here.

pub struct GTD2<S: ?Sized, P: Projector<S>> {
    pub fa_theta: SimpleLinear<S, P>,
    pub fa_w: SimpleLinear<S, P>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S: ?Sized, P: Projector<S>> GTD2<S, P> {
    pub fn new<T1, T2, T3>(
        fa_theta: SimpleLinear<S, P>,
        fa_w: SimpleLinear<S, P>,
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

impl<S, P: Projector<S>> Handler for GTD2<S, P> {
    type Sample = Transition<S, ()>;

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let phi_s = self.fa_theta.projector.project(&sample.from.state());
        let phi_ns = self.fa_theta.projector.project(&sample.to.state());

        let td_error = sample.reward + self.gamma * self.fa_theta.evaluate_phi(&phi_ns)
            - self.fa_theta.evaluate_phi(&phi_s);
        let td_estimate = self.fa_w.evaluate_phi(&phi_s);

        let span = self.fa_theta.projector.span();
        let update = phi_s.clone().expanded(span) - self.gamma.value() * phi_ns.expanded(span);

        self.fa_w
            .update_phi(&phi_s, self.beta * (td_error - td_estimate));
        self.fa_theta
            .update_phi(&Projection::Dense(update), self.alpha * td_estimate);
    }

    fn handle_terminal(&mut self, _: &Self::Sample) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, P: Projector<S>> Predictor<S> for GTD2<S, P> {
    fn evaluate(&self, s: &S) -> f64 { self.fa_theta.evaluate(s).unwrap() }
}

pub struct TDC<S: ?Sized, P: Projector<S>> {
    pub fa_theta: SimpleLinear<S, P>,
    pub fa_w: SimpleLinear<S, P>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S: ?Sized, P: Projector<S>> TDC<S, P> {
    pub fn new<T1, T2, T3>(
        fa_theta: SimpleLinear<S, P>,
        fa_w: SimpleLinear<S, P>,
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

impl<S, P: Projector<S>> Handler for TDC<S, P> {
    type Sample = Transition<S, ()>;

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let phi_s = self.fa_theta.projector.project(&sample.from.state());
        let phi_ns = self.fa_theta.projector.project(&sample.to.state());

        let td_error = sample.reward + self.gamma * self.fa_theta.evaluate_phi(&phi_ns)
            - self.fa_theta.evaluate_phi(&phi_s);
        let td_estimate = self.fa_w.evaluate_phi(&phi_s);

        let span = self.fa_theta.projector.span();
        let update = td_error * phi_s.clone().expanded(span)
            - self.gamma.value() * td_estimate * &phi_ns.expanded(span);

        self.fa_w
            .update_phi(&phi_s, self.beta * (td_error - td_estimate));
        self.fa_theta
            .update_phi(&Projection::Dense(update), self.alpha.value());
    }

    fn handle_terminal(&mut self, _: &Self::Sample) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, P: Projector<S>> Predictor<S> for TDC<S, P> {
    fn evaluate(&self, s: &S) -> f64 { self.fa_theta.evaluate(s).unwrap() }
}

// TODO:
// ABQ(lambda) - https://arxiv.org/pdf/1702.03006.pdf
// GQ(lambda) - http://agi-conf.org/2010/wp-content/uploads/2009/06/paper_21.pdf
// GTD(lambda) - https://era.library.ualberta.ca/files/8s45q967t/Hamid_Maei_PhDThesis.pdf
// True online GTD(lambda) - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.487.2451&rep=rep1&type=pdf
// GTD2(lambda)-MP - https://arxiv.org/pdf/1602.08771.pdf
// TDC(lambda)-MP - https://arxiv.org/pdf/1602.08771.pdf
