use std::fmt;

#[derive(Debug)]
pub enum Error {
    MissingMeanData(String),
    MissingStdData(String),
    UnconnectedBlocks,
    LastTsMissing,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::MissingMeanData(s) => write!(f, "Missing mean field {} ", s),
            Error::MissingStdData(s) => write!(f, "Missing std field {} ", s),
            Error::UnconnectedBlocks => write!(f, "Supplied blocks must be ordered and connected "),
            Error::LastTsMissing => write!(f, "None of the 10 blocks is"),
        }
    }
}

impl std::error::Error for Error {}
