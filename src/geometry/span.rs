use std::ops::Mul;


#[derive(Clone, Copy, Debug)]
pub enum Span {
    Null,
    Finite(usize),
    Infinite,
}

impl Mul for Span {
    type Output = Span;

    fn mul(self, rhs: Span) -> Span {
        match self {
            Span::Null => rhs,
            Span::Infinite => self,
            Span::Finite(ls) => {
                match rhs {
                    Span::Null => self,
                    Span::Infinite => rhs,
                    Span::Finite(rs) => Span::Finite(ls * rs),
                }
            }
        }
    }
}

impl Into<usize> for Span {
    fn into(self) -> usize {
        match self {
            Span::Finite(e) => e,
            _ => panic!("Span type has no integer representation.")
        }
    }
}
