use std::fmt;
use std::marker::PhantomData;

use serde::de::{Deserialize, Deserializer, Error, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeSeq, Serializer};

use crate::matrix::{Matrix, SizeMarker};

struct MatrixVisitor<W: SizeMarker, H: SizeMarker> {
    marker: PhantomData<fn() -> Matrix<W, H>>,
}

impl<W: SizeMarker, H: SizeMarker> MatrixVisitor<W, H> {
    fn new() -> Self {
        MatrixVisitor {
            marker: PhantomData,
        }
    }
}

impl<'de, W: SizeMarker, H: SizeMarker> Visitor<'de> for MatrixVisitor<W, H> {
    type Value = Matrix<W, H>;

    // Format a message stating what data this Visitor expects to receive.
    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Matrix of f32")
    }

    fn visit_seq<S>(self, mut access: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let mut buffer = Vec::with_capacity(H::size() * W::size());

        let mut fetch_row = |v: Vec<f32>, row| -> Result<(), S::Error> {
            if v.len() != W::size() {
                return Err(<S as SeqAccess<'de>>::Error::custom(format!(
                    "Invalid matrix width at row {}: expected {}, found {}",
                    row,
                    W::size(),
                    v.len()
                )));
            }
            buffer.extend(v);

            Ok(())
        };

        if H::size() == 1 {
            // Matrix with height 1, encoded as a vector
            let mut v = Vec::with_capacity(W::size());
            while let Some(f) = access.next_element()? {
                v.push(f);
            }

            fetch_row(v, 0)?;
        } else {
            let mut rows = 0;
            while let Some(v) = access.next_element::<Vec<f32>>()? {
                fetch_row(v, rows)?;
                rows += 1;
            }
            if rows != H::size() {
                return Err(<S as SeqAccess<'de>>::Error::custom(format!(
                    "Invalid matrix height: expected {}, found {}",
                    H::size(),
                    rows
                )));
            }
        }

        dbg!((W::size(), H::size()));

        Ok(Matrix::from_buffer(buffer.into_boxed_slice()))
    }
}

impl<'de, W: SizeMarker, H: SizeMarker> Deserialize<'de> for Matrix<W, H> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(MatrixVisitor::new())
    }
}

impl<W: SizeMarker, H: SizeMarker> Serialize for Matrix<W, H> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(H::size()))?;
        for i in 0..H::size() {
            seq.serialize_element(&self[i])?;
        }

        seq.end()
    }
}
