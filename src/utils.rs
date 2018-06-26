#![allow(dead_code)]
use std::f64;

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

pub fn sub2ind(dims: &[usize], inds: &[usize]) -> usize {
    let d_it = dims.iter().rev().skip(1);
    let i_it = inds.iter().rev().skip(1);

    d_it.zip(i_it)
        .fold(inds.last().cloned().unwrap(), |acc, (d, i)| i + d * acc)
}

/// Given a vector containing a partial Cartesian product, and a list of items,
/// return a vector adding the list of items to the partial Cartesian product.
///
// /// # Example
// ///
// /// ```
// /// use rsrl::utils::partial_cartesian;
// ///
// /// let partial_product = vec![vec![1, 4], vec![1, 5], vec![2, 4], vec![2, 5]];
// /// let items = vec![6, 7];
// /// let next_product = partial_cartesian(partial_product, &items);
// ///
// /// assert_eq!(
// ///     next_product,
// ///     vec![
// ///         vec![1, 4, 6],
// ///         vec![1, 4, 7],
// ///         vec![1, 5, 6],
// ///         vec![1, 5, 7],
// ///         vec![2, 4, 6],
// ///         vec![2, 4, 7],
// ///         vec![2, 5, 6],
// ///         vec![2, 5, 7],
// ///     ]
// /// );
// /// ```
///
/// Pulled from [here](https://gist.github.com/kylewlacy/115965b40e02a3325558).
pub fn partial_cartesian<T: Clone>(a: Vec<Vec<T>>, b: &Vec<T>) -> Vec<Vec<T>> {
    a.into_iter()
        .flat_map(|xs| {
            b.iter()
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

/// Computes the Cartesian product of lists[0] * lists[1] * ... * lists[n].
///
// /// # Example
// ///
// /// ```
// /// use rsrl::utils::cartesian_product;
// ///
// /// let lists = vec![vec![1, 2], vec![4, 5], vec![6, 7]];
// /// let product = cartesian_product(&lists);
// ///
// /// assert_eq!(
// ///     product,
// ///     vec![
// ///         vec![1, 4, 6],
// ///         vec![1, 4, 7],
// ///         vec![1, 5, 6],
// ///         vec![1, 5, 7],
// ///         vec![2, 4, 6],
// ///         vec![2, 4, 7],
// ///         vec![2, 5, 6],
// ///         vec![2, 5, 7],
// ///     ]
// /// );
// /// ```
///
/// Pulled from [here](https://gist.github.com/kylewlacy/115965b40e02a3325558).
pub fn cartesian_product<T: Clone>(lists: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    match lists.split_first() {
        Some((first, rest)) => {
            let init: Vec<Vec<T>> = first.iter().cloned().map(|n| vec![n]).collect();

            rest.iter()
                .cloned()
                .fold(init, |vec, list| partial_cartesian(vec, &list))
        },
        None => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::{cartesian_product, sub2ind};

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
    fn test_sub2ind_empty() { sub2ind(&vec![], &vec![]); }

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
