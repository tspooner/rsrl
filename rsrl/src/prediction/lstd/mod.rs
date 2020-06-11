// TODO: Implement regularized LSTD "http://mlg.eng.cam.ac.uk/hoffmanm/papers/hoffman:2012b.pdf
pub mod lstd;
pub mod ilstd;
pub mod lstd_lambda;
pub mod lambda_lspe;
pub mod recursive_lstd;

pub use self::{
    lstd::LSTD,
    ilstd::iLSTD,
    lstd_lambda::LSTDLambda,
    lambda_lspe::LambdaLSPE,
    recursive_lstd::RecursiveLSTD,
};
