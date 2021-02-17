use crate::matrix::Size;
use std::fmt;

#[derive(Debug)]
pub enum Error {
    IncompatibleDotMatrix(Size, Size),
    IncompatibleAddMatrix(Size, Size),
    MissingMeanData(String),
    MissingStdData(String),
    Serde(serde_cbor::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::IncompatibleDotMatrix(a1, a2) => write!(
                f,
                "Matrix sizes are incompatible for dot product {:?} {:?} ",
                a1, a2
            ),
            Error::IncompatibleAddMatrix(a1, a2) => write!(
                f,
                "Matrix sizes are incompatible for adding {:?} {:?} ",
                a1, a2
            ),
            Error::MissingMeanData(s) => write!(f, "Missing mean field {} ", s),
            Error::MissingStdData(s) => write!(f, "Missing std field {} ", s),
            Error::Serde(e) => write!(f, "Serde {:?} ", e),
        }
    }
}

impl std::error::Error for Error {}

impl From<serde_cbor::Error> for Error {
    fn from(e: serde_cbor::Error) -> Self {
        Error::Serde(e)
    }
}
