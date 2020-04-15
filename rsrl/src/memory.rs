use std::{cell::{RefCell, Ref, RefMut}, fmt, ops::Deref, rc::Rc};

pub fn make_shared<T>(t: T) -> Shared<T> { Shared(Rc::new(RefCell::new(t))) }

pub struct Shared<T>(pub Rc<RefCell<T>>);

impl<T> Shared<T> {
    pub fn new(t: T) -> Shared<T> {
        make_shared(t)
    }

    pub fn borrow(&self) -> Ref<T> {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<T> {
        self.0.borrow_mut()
    }

    pub fn as_ptr(&self) -> *mut T {
        self.0.as_ptr()
    }
}

impl<T: fmt::Display> fmt::Display for Shared<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.deref())
    }
}

impl<T: fmt::Debug> fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.deref())
    }
}

impl<'a, T> Deref for Shared<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { self.as_ptr().as_ref().unwrap() }
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Shared<T> {
        Shared(self.0.clone())
    }
}
