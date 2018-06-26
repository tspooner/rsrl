use std::cell::RefCell;
use std::rc::Rc;

pub type Shared<T> = Rc<RefCell<T>>;

pub fn make_shared<T>(t: T) -> Shared<T> { Rc::new(RefCell::new(t)) }
