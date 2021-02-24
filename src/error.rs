use std::fmt;

#[derive(Debug)]
pub enum Error {
    MissingMeanData(String),
    MissingStdData(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::MissingMeanData(s) => write!(f, "Missing mean field {} ", s),
            Error::MissingStdData(s) => write!(f, "Missing std field {} ", s),
        }
    }
}

impl std::error::Error for Error {}
