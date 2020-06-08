extern crate proc_macro;
#[macro_use]
extern crate quote;
#[macro_use]
extern crate syn;

mod parameterised;

#[proc_macro_derive(Parameterised, attributes(weights))]
pub fn derive_parameterised(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse2(tokens.into()).unwrap();

    parameterised::expand_derive_parameterised(&ast).into()
}
