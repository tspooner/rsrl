use agents::Predictor;
use domains::Transition;
use fa::{Approximator, Projection, Projector, SimpleLFA, VFunction};
use {Handler, Parameter};

// TODO: Implement TDPredictor for all agents here.

pub struct GTD2<S: ?Sized, P: Projector<S>> {
    pub fa_theta: SimpleLFA<S, P>,
    pub fa_w: SimpleLFA<S, P>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S: ?Sized, P: Projector<S>> GTD2<S, P> {
    pub fn new<T1, T2, T3>(
        fa_theta: SimpleLFA<S, P>,
        fa_w: SimpleLFA<S, P>,
        alpha: T1,
        beta: T2,
        gamma: T3,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        if !(fa_theta.projector.dim() == fa_w.projector.dim()) {
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

impl<S, P: Projector<S>> Handler<Transition<S, ()>> for GTD2<S, P> {
    fn handle_sample(&mut self, sample: &Transition<S, ()>) {
        let phi_s = self.fa_theta.projector.project(&sample.from.state());
        let phi_ns = self.fa_theta.projector.project(&sample.to.state());

        let td_error = sample.reward + self.gamma * self.fa_theta.evaluate_phi(&phi_ns)
            - self.fa_theta.evaluate_phi(&phi_s);
        let td_estimate = self.fa_w.evaluate_phi(&phi_s);

        let dim = self.fa_theta.projector.dim();
        let update = phi_s.clone().expanded(dim) - self.gamma.value() * phi_ns.expanded(dim);

        self.fa_w
            .update_phi(&phi_s, self.beta * (td_error - td_estimate));
        self.fa_theta
            .update_phi(&Projection::Dense(update), self.alpha * td_estimate);
    }

    fn handle_terminal(&mut self, _: &Transition<S, ()>) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, P: Projector<S>> Predictor<S> for GTD2<S, P> {
    fn predict(&mut self, s: &S) -> f64 { self.fa_theta.evaluate(s).unwrap() }
}

pub struct TDC<S: ?Sized, P: Projector<S>> {
    pub fa_theta: SimpleLFA<S, P>,
    pub fa_w: SimpleLFA<S, P>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S: ?Sized, P: Projector<S>> TDC<S, P> {
    pub fn new<T1, T2, T3>(
        fa_theta: SimpleLFA<S, P>,
        fa_w: SimpleLFA<S, P>,
        alpha: T1,
        beta: T2,
        gamma: T3,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        if !(fa_theta.projector.dim() == fa_w.projector.dim()) {
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

impl<S, P: Projector<S>> Handler<Transition<S, ()>> for TDC<S, P> {
    fn handle_sample(&mut self, sample: &Transition<S, ()>) {
        let phi_s = self.fa_theta.projector.project(&sample.from.state());
        let phi_ns = self.fa_theta.projector.project(&sample.to.state());

        let td_error = sample.reward + self.gamma * self.fa_theta.evaluate_phi(&phi_ns)
            - self.fa_theta.evaluate_phi(&phi_s);
        let td_estimate = self.fa_w.evaluate_phi(&phi_s);

        let dim = self.fa_theta.projector.dim();
        let update = td_error * phi_s.clone().expanded(dim)
            - self.gamma.value() * td_estimate * &phi_ns.expanded(dim);

        self.fa_w
            .update_phi(&phi_s, self.beta * (td_error - td_estimate));
        self.fa_theta
            .update_phi(&Projection::Dense(update), self.alpha.value());
    }

    fn handle_terminal(&mut self, _: &Transition<S, ()>) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, P: Projector<S>> Predictor<S> for TDC<S, P> {
    fn predict(&mut self, s: &S) -> f64 { self.fa_theta.evaluate(s).unwrap() }
}

// TODO:
// ABQ(lambda) - https://arxiv.org/pdf/1702.03006.pdf
// GQ(lambda) - http://agi-conf.org/2010/wp-content/uploads/2009/06/paper_21.pdf
// GTD(lambda) - https://era.library.ualberta.ca/files/8s45q967t/Hamid_Maei_PhDThesis.pdf
// True online GTD(lambda) - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.487.2451&rep=rep1&type=pdf
// GTD2(lambda)-MP - https://arxiv.org/pdf/1602.08771.pdf
// TDC(lambda)-MP - https://arxiv.org/pdf/1602.08771.pdf
