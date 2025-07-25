// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]

mod error;
pub mod interpreter;
pub mod minimal_bindings;
pub mod model;
pub mod tensor;

// Re-export the minimal bindings for direct use if needed
pub use minimal_bindings::*;

pub use self::error::{Error, ErrorKind, Result};
