extern crate blas_sys;
use self::blas_sys::c::cblas_ddot;
use std::cmp::min;

use std::f64;


pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    let n: i32 = min(x.len() as i32, y.len() as i32);

    unsafe { cblas_ddot(n, x.as_ptr(), 1, y.as_ptr(), 1) }
}


pub fn argmaxima(vals: &[f64]) -> (f64, Vec<usize>) {
    let mut max = f64::MIN;
    let mut ixs = vec![];

    for (i, &v) in vals.iter().enumerate() {
        if (v - max).abs() < 1e-7 {
            ixs.push(i);
        } else if v > max {
            max = v;
            ixs.clear();
            ixs.push(i);
        }
    }

    (max, ixs)
}


// TODO: Pass by iterator so that we don't have to collect before passing.
pub fn sub2ind(dims: &[usize], inds: &[usize]) -> usize {
    let d_it = dims.iter().rev().skip(1);
    let i_it = inds.iter().rev().skip(1);

    d_it.zip(i_it).fold(inds.last().cloned().unwrap(), |acc, (d, i)| i + d * acc)
}


pub fn cartesian_product<T: Clone>(lists: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    fn partial_cartesian<T: Clone>(u: Vec<Vec<T>>, v: &Vec<T>) -> Vec<Vec<T>> {
        u.into_iter()
            .flat_map(|xs| {
                v.iter()
                    .cloned()
                    .map(|y| {
                        let mut vec = xs.clone();
                        vec.push(y);
                        vec
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    match lists.split_first() {
        Some((head, tail)) => {
            let init: Vec<Vec<T>> = head.iter().cloned().map(|n| vec![n]).collect();

            tail.iter().cloned().fold(init, |vec, list| partial_cartesian(vec, &list))
        }
        None => vec![],
    }
}


#[cfg(test)]
mod tests {
    use super::{sub2ind, cartesian_product};

    #[test]
    fn test_sub2ind() {
        assert_eq!(sub2ind(&vec![5, 7, 12], &vec![3, 0, 0]), 3);
        assert_eq!(sub2ind(&vec![5, 7, 12], &vec![2, 0, 5]), 177);
        assert_eq!(sub2ind(&vec![5, 7, 12], &vec![2, 5, 5]), 202);
        assert_eq!(sub2ind(&vec![5, 7, 12], &vec![0, 0, 11]), 385);
        assert_eq!(sub2ind(&vec![5, 7, 12], &vec![5, 0, 11]), 390);
    }

    #[test]
    #[should_panic]
    fn test_sub2ind_empty() {
        sub2ind(&vec![], &vec![]);
    }


    #[test]
    fn test_cartesian_product() {
        let to_combine = vec![vec![0.0, 1.0, 2.0], vec![7.0, 8.0, 9.0]];

        let combs = cartesian_product(&to_combine);

        assert_eq!(combs[0], vec![0.0, 7.0]);
        assert_eq!(combs[1], vec![0.0, 8.0]);
        assert_eq!(combs[2], vec![0.0, 9.0]);

        assert_eq!(combs[3], vec![1.0, 7.0]);
        assert_eq!(combs[4], vec![1.0, 8.0]);
        assert_eq!(combs[5], vec![1.0, 9.0]);

        assert_eq!(combs[6], vec![2.0, 7.0]);
        assert_eq!(combs[7], vec![2.0, 8.0]);
        assert_eq!(combs[8], vec![2.0, 9.0]);
    }
}
