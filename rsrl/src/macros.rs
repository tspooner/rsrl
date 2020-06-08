#![macro_use]

#[allow(unused_macros)]
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

#[allow(unused_macros)]
macro_rules! clip {
    ($lb:expr, $x:expr, $ub:expr) => {{
        $lb.max($ub.min($x))
    }};
}

macro_rules! import_all {
    ($module:ident) => {
        mod $module;
        pub use self::$module::*;
    };
}
