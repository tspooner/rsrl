use std::{
    cell::{Ref, RefCell, RefMut},
    fmt,
    ops::{Deref, Index},
    rc::Rc,
};

pub fn make_shared<T>(t: T) -> Shared<T> { Shared(Rc::new(RefCell::new(t))) }

pub struct Shared<T>(pub Rc<RefCell<T>>);

impl<T> Shared<T> {
    pub fn new(t: T) -> Shared<T> { make_shared(t) }

    pub fn borrow(&self) -> Ref<T> { self.0.borrow() }

    pub fn borrow_mut(&self) -> RefMut<T> { self.0.borrow_mut() }

    pub fn as_ptr(&self) -> *mut T { self.0.as_ptr() }
}

impl<T: fmt::Display> fmt::Display for Shared<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", self.deref()) }
}

impl<T: fmt::Debug> fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{:?}", self.deref()) }
}

impl<'a, T> Deref for Shared<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T { unsafe { self.as_ptr().as_ref().unwrap() } }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Shared<T> { Shared(self.0.clone()) }
}

pub type OutputOf<F, S> = <F as Function<S>>::Output;

// TODO: When variadic generics drop, replace Args tuple with sequence of
// inputs...
pub trait Function<Args> {
    type Output;

    fn evaluate(&self, args: Args) -> Self::Output;
}

impl<Args, F: Function<Args>> Function<Args> for Shared<F> {
    type Output = F::Output;

    fn evaluate(&self, args: Args) -> Self::Output { self.borrow().evaluate(args) }
}

impl<F, S, O> Function<S> for F
where F: Fn(S) -> O
{
    type Output = O;

    fn evaluate(&self, state: S) -> Self::Output { (self)(state) }
}

// TODO: Generalise this to functions with an argmax.
pub trait Enumerable<Args>: Function<Args>
where
    Self::Output: Index<usize> + IntoIterator<Item = <Self::Output as Index<usize>>::Output>,

    <Self::Output as Index<usize>>::Output: Sized,
    <Self::Output as IntoIterator>::IntoIter: ExactSizeIterator,
{
    fn len(&self, args: Args) -> usize { self.evaluate(args).into_iter().len() }

    fn evaluate_index(&self, args: Args, index: usize) -> <Self::Output as Index<usize>>::Output {
        let val: *const <Self::Output as Index<usize>>::Output = &self.evaluate(args)[index];

        unsafe { val.read() }
    }

    fn find_min(&self, args: Args) -> (usize, <Self::Output as Index<usize>>::Output)
    where
        Self::Output: IntoIterator<Item = <Self::Output as Index<usize>>::Output>,
        <Self::Output as Index<usize>>::Output: PartialOrd + Sized,
    {
        let mut iter = self.evaluate(args).into_iter().enumerate();
        let first = iter.next().unwrap();

        iter.fold(first, |acc, (i, x)| if acc.1 < x { acc } else { (i, x) })
    }

    fn find_max(&self, args: Args) -> (usize, <Self::Output as Index<usize>>::Output)
    where
        Self::Output: IntoIterator<Item = <Self::Output as Index<usize>>::Output>,
        <Self::Output as Index<usize>>::Output: PartialOrd + Sized,
    {
        let mut iter = self.evaluate(args).into_iter().enumerate();
        let first = iter.next().unwrap();

        iter.fold(first, |acc, (i, x)| if acc.1 > x { acc } else { (i, x) })
    }
}

impl<Args, F: Enumerable<Args>> Enumerable<Args> for Shared<F>
where
    F::Output: Index<usize> + IntoIterator<Item = <F::Output as Index<usize>>::Output>,

    <Self::Output as Index<usize>>::Output: Sized,
    <Self::Output as IntoIterator>::IntoIter: ExactSizeIterator,
{
}

impl<F, S, O> Enumerable<S> for F
where
    F: Fn(S) -> O,
    O: Index<usize> + IntoIterator<Item = <O as Index<usize>>::Output>,

    <O as Index<usize>>::Output: Sized,
    <O as IntoIterator>::IntoIter: ExactSizeIterator,
{
}

pub trait Differentiable<Args>: Function<Args> + crate::params::Parameterised {
    type Jacobian: crate::params::BufferMut;

    fn grad(&self, args: Args) -> Self::Jacobian;

    fn grad_log(&self, args: Args) -> Self::Jacobian;
}

impl<Args, F: Differentiable<Args>> Differentiable<Args> for Shared<F> {
    type Jacobian = F::Jacobian;

    fn grad(&self, args: Args) -> Self::Jacobian { self.borrow().grad(args) }

    fn grad_log(&self, args: Args) -> Self::Jacobian { self.borrow().grad_log(args) }
}

pub trait Handler<M> {
    type Response;
    type Error;

    fn handle(&mut self, msg: M) -> Result<Self::Response, Self::Error>;

    fn handle_unchecked(&mut self, msg: M) -> Self::Response { self.handle(msg).ok().unwrap() }
}

impl<M, T: Handler<M>> Handler<M> for Shared<T> {
    type Response = T::Response;
    type Error = T::Error;

    fn handle(&mut self, msg: M) -> Result<Self::Response, Self::Error> {
        self.borrow_mut().handle(msg)
    }

    fn handle_unchecked(&mut self, msg: M) -> Self::Response {
        self.borrow_mut().handle_unchecked(msg)
    }
}
