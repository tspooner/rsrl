use proc_macro2::TokenStream;
use quote::ToTokens;
use std::iter;
use syn::{Data, DataStruct, Field, Fields, Generics, Ident, Meta, Type};

const WEIGHTS: &str = "weights";

struct Implementation {
    pub generics: Option<Vec<Type>>,
    pub body: Body,
}

impl Implementation {
    pub fn concrete(body: Body) -> Implementation {
        Implementation {
            generics: None,
            body,
        }
    }

    pub fn with_generics(generics: Vec<Type>, body: Body) -> Implementation {
        Implementation {
            generics: Some(generics),
            body,
        }
    }

    fn make_where_predicates(types: &[Type]) -> Vec<syn::WherePredicate> {
        types
            .iter()
            .map(|g| parse_quote! { #g: crate::params::Parameterised })
            .collect()
    }

    pub fn add_trait_bounds(&self, mut generics: Generics) -> Generics {
        if let Some(ref gs) = self.generics {
            let new_ps = Self::make_where_predicates(gs);

            generics.make_where_clause().predicates.extend(new_ps);
        }

        generics
    }
}

struct Body {
    pub weights: Option<TokenStream>,
    pub weights_dim: Option<TokenStream>,
    pub weights_view: TokenStream,
    pub weights_view_mut: TokenStream,
}

impl ToTokens for Body {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if let Some(ref weights_fn) = self.weights {
            tokens.extend(iter::once(quote! {
                fn weights(&self) -> crate::params::Weights { #weights_fn }
            }));
        }

        if let Some(ref weights_dim_fn) = self.weights_dim {
            tokens.extend(iter::once(quote! {
                fn weights_dim(&self) -> (usize, usize) { #weights_dim_fn }
            }));
        }

        let weights_view_fn = &self.weights_view;
        let weights_view_mut_fn = &self.weights_view_mut;

        tokens.extend(
            iter::once(quote! {
                fn weights_view(&self) -> crate::params::WeightsView {
                    #weights_view_fn
                }
            })
            .chain(iter::once(quote! {
                fn weights_view_mut(&mut self) -> crate::params::WeightsViewMut {
                    #weights_view_mut_fn
                }
            })),
        );
    }
}

struct WeightsField<'a, I: ToTokens> {
    pub ident: I,
    pub field: &'a Field,
}

impl<'a, I: ToTokens> WeightsField<'a, I> {
    pub fn type_ident(&self) -> &'a Ident {
        match self.field.ty {
            Type::Path(ref tp) => &tp.path.segments[0].ident,
            _ => unimplemented!(),
        }
    }
}

pub fn expand_derive_parameterised(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let implementation = parameterised_impl(&ast.data);

    let body = &implementation.body;
    let generics = implementation.add_trait_bounds(ast.generics.clone());
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote! {
        #[automatically_derived]
        impl #impl_generics crate::params::Parameterised for #name #ty_generics #where_clause {
            #body
        }
    }
}

fn parameterised_impl(data: &Data) -> Implementation {
    match data {
        Data::Struct(ref ds) => parameterised_struct_impl(ds),
        _ => unimplemented!(),
    }
}

fn parameterised_struct_impl(ds: &DataStruct) -> Implementation {
    let n_fields = ds.fields.iter().len();

    if n_fields > 1 {
        let mut annotated_fields: Vec<_> = ds
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| has_weight_attribute(f))
            .map(|(i, f)| WeightsField {
                ident: f
                    .ident
                    .clone()
                    .map(|i| quote! { #i })
                    .unwrap_or(quote! { #i }),
                field: f,
            })
            .collect();

        if annotated_fields.is_empty() {
            let (index, iwf) = ds
                .fields
                .iter()
                .enumerate()
                .find(|(_, f)| match &f.ident {
                    Some(ident) => *ident == WEIGHTS,
                    None => false,
                })
                .expect("Couldn't infer weights field, consider annotating with #[weights].");

            parameterised_wf_impl(WeightsField {
                ident: iwf
                    .ident
                    .clone()
                    .map(|i| quote! { #i })
                    .unwrap_or(quote! { #index }),
                field: iwf,
            })
        } else if annotated_fields.len() == 1 {
            parameterised_wf_impl(annotated_fields.pop().unwrap())
        } else {
            panic!(
                "Duplicate #[weights] annotations - \
                 automatic view concatenation implementations are not currently supported."
            )
        }
    } else if n_fields == 1 {
        match ds.fields {
            Fields::Unnamed(ref fs) => parameterised_wf_impl(WeightsField {
                ident: quote! { 0 },
                field: &fs.unnamed[0],
            }),
            Fields::Named(ref fs) => parameterised_wf_impl(WeightsField {
                ident: fs.named[0].ident.clone(),
                field: &fs.named[0],
            }),
            _ => unreachable!(),
        }
    } else {
        panic!("Nothing to derive Parameterised from!")
    }
}

fn parameterised_wf_impl<I: ToTokens>(wf: WeightsField<I>) -> Implementation {
    let ident = &wf.ident;
    let type_ident = wf.type_ident();

    match type_ident.to_string().as_ref() {
        "Vec" => Implementation::concrete(Body {
            weights: Some(quote! {
                let n_rows = self.#ident.len();

                crate::params::Weights::from_shape_vec((n_rows, 1), self.#ident.clone()).unwrap()
            }),
            weights_dim: Some(quote! { (self.#ident.len(), 1) }),
            weights_view: quote! {
                let n_rows = self.#ident.len();

                crate::params::WeightsView::from_shape((n_rows, 1), &self.#ident).unwrap()
            },
            weights_view_mut: quote! {
                let n_rows = self.#ident.len();

                crate::params::WeightsViewMut::from_shape((n_rows, 1), &mut self.#ident).unwrap()
            },
        }),
        "Array1" => Implementation::concrete(Body {
            weights: Some(quote! {
                let n_rows = self.#ident.len();

                self.#ident.clone().into_shape((n_rows, 1)).unwrap()
            }),
            weights_dim: Some(quote! { (self.#ident.len(), 1) }),
            weights_view: quote! {
                let n_rows = self.#ident.len();

                self.#ident.view().into_shape((n_rows, 1)).unwrap()
            },
            weights_view_mut: quote! {
                let n_rows = self.#ident.len();

                self.#ident.view_mut().into_shape((n_rows, 1)).unwrap()
            },
        }),
        "Weights" | "Array2" => Implementation::concrete(Body {
            weights: Some(quote! { self.#ident.clone() }),
            weights_dim: Some(quote! { self.#ident.dim() }),
            weights_view: quote! { self.#ident.view() },
            weights_view_mut: quote! { self.#ident.view_mut() },
        }),
        _ => Implementation::with_generics(
            vec![wf.field.ty.clone()],
            Body {
                weights: Some(quote! { self.#ident.weights() }),
                weights_dim: Some(quote! { self.#ident.weights_dim() }),
                weights_view: quote! { self.#ident.weights_view() },
                weights_view_mut: quote! { self.#ident.weights_view_mut() },
            },
        ),
    }
}

fn has_weight_attribute(f: &Field) -> bool {
    f.attrs.iter().any(|a| {
        a.parse_meta()
            .map(|meta| match meta {
                Meta::Path(ref path) => path.is_ident(WEIGHTS),
                _ => false,
            })
            .unwrap_or(false)
    })
}
