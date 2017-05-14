#![macro_use]

macro_rules! clip {
    ($lb:expr, $x:expr, $ub:expr) => {
        {
            $lb.max($ub.min($x))
        }
    }
}
