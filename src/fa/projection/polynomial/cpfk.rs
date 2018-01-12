#![allow(dead_code)]

pub(super) fn t_0(_x: f64) -> f64 {
    1.0
}

pub(super) fn t_1(x: f64) -> f64 {
    x
}

pub(super) fn t_2(x: f64) -> f64 {
    2.0*x*x - 1.0
}

pub(super) fn t_3(x: f64) -> f64 {
    4.0*x*x*x - 3.0*x
}

pub(super) fn t_4(x: f64) -> f64 {
    8.0*x*x*x*x - 8.0*x*x + 1.0
}

pub(super) fn t_5(x: f64) -> f64 {
    16.0*x.powi(5) - 20.0*x.powi(3) + 5.0*x
}

pub(super) fn t_6(x: f64) -> f64 {
    32.0*x.powi(6) - 48.0*x.powi(4) + 18.0*x.powi(2) - 1.0
}

pub(super) fn t_7(x: f64) -> f64 {
    64.0*x.powi(7) - 112.0*x.powi(5) + 56.0*x.powi(3) - 7.0*x
}

pub(super) fn t_8(x: f64) -> f64 {
    128.0*x.powi(8) - 256.0*x.powi(6) + 160.0*x.powi(4) - 32.0*x.powi(2) + 1.0
}

pub(super) fn t_9(x: f64) -> f64 {
    256.0*x.powi(9) - 576.0*x.powi(7) + 432.0*x.powi(5) - 120.0*x.powi(3) + 9.0*x
}

pub(super) fn t_10(x: f64) -> f64 {
    512.0*x.powi(10) - 1280.0*x.powi(8) + 1120.0*x.powi(6) - 400.0*x.powi(4) + 50.0*x.powi(2) - 1.0
}

pub(super) fn t_11(x: f64) -> f64 {
    1024.0*x.powi(11) - 2816.0*x.powi(9) + 2816.0*x.powi(7) - 1232.0*x.powi(5) + 220.0*x.powi(3)
        - 11.0*x
}
