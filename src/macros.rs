#![macro_use]

macro_rules! wrap {
    ($lb:expr, $x:expr, $ub:expr) => {{
        let mut nx = $x;
        let diff = $ub - $lb;

        while nx > $ub {
            nx -= diff;
        }

        while nx < $lb {
            nx += diff;
        }

        nx
    }};
}

macro_rules! clip {
    ($lb:expr, $x:expr, $ub:expr) => {{
        $lb.max($ub.min($x))
    }};
}
