// Semi-gradient methods:
pub mod td;
pub mod td_lambda;

pub use self::{td::TD, td_lambda::TDLambda};

// Full-gradient methods:
pub mod gtd2;
pub mod tdc;

pub use self::{gtd2::GTD2, tdc::TDC};

// TODO:
// n-step TD - Sutton & Barto
// ETD(lambda) - https://arxiv.org/pdf/1503.04269.pdf
// HTD(lambda) - https://arxiv.org/pdf/1602.08771.pdf
// PTD(lambda) - http://proceedings.mlr.press/v32/sutton14.pdf
// True online TD(lambda) - http://proceedings.mlr.press/v32/seijen14.pdf
// True online ETD(lambda) - https://arxiv.org/pdf/1602.08771.pdf
// True online ETD(beta, lambda) - https://arxiv.org/pdf/1602.08771.pdf
// True online HTD(lambda) - https://arxiv.org/pdf/1602.08771.pdf
