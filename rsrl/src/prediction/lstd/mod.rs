// TODO: Implement regularized LSTD "http://mlg.eng.cam.ac.uk/hoffmanm/papers/hoffman:2012b.pdf
pub mod ilstd;
pub mod lambda_lspe;
pub mod lstd;
pub mod lstd_lambda;
pub mod recursive_lstd;

pub use self::{
    ilstd::iLSTD,
    lambda_lspe::LambdaLSPE,
    lstd::LSTD,
    lstd_lambda::LSTDLambda,
    recursive_lstd::RecursiveLSTD,
};
