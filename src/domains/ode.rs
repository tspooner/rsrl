use crate::core::Vector;
use ndarray::{Axis, Array2};

pub(crate) fn runge_kutta4(fx: &Fn(&Vector) -> Vector, y: Vector, times: Vector) -> Vector {
    // more or less copied from python version
    let mut yout = Array2::zeros((times.len(), y.len()));
    yout.row_mut(0).assign(&y);
    for i in 0..(times.len()-1) {
        // let t = times[i];
        let dt = times[i+1] - times[i];
        let dt2 = dt / 2.0;
        let y0 = yout.row(i).to_owned();

        let k1 = fx(&y0);
        let k2 = fx(&(y0.clone() + dt2*k1.clone()));
        let k3 = fx(&(y0.clone() + dt2 * k2.clone()));
        let k4 = fx(&(y0.clone() + dt * k3.clone()));

        yout.row_mut(i+1).assign(&(y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)));
    }
    yout.row(yout.len_of(Axis(0))-1).to_owned()
}
