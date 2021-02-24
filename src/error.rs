use std::fmt;

#[derive(Debug)]
pub enum Error {
    MissingMeanData(String),
    MissingStdData(String),
    Serde(serde_cbor::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
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
