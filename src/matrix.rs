use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub mod size {
    include!(concat!(env!("OUT_DIR"), "/sizes.rs"));
}

use size::*;

pub trait SizeMarker: std::fmt::Debug {
    fn size() -> usize;
}

#[derive(Debug)]
pub struct Matrix<W, H>(Box<[f32]>, PhantomData<W>, PhantomData<H>);

impl<W: SizeMarker, H: SizeMarker> Default for Matrix<W, H> {
    fn default() -> Self {
        let slice = vec![0.0; W::size() * H::size()].into_boxed_slice();

        Matrix(slice, PhantomData, PhantomData)
    }
}

impl<W: SizeMarker, H: SizeMarker> Matrix<W, H> {
    pub(crate) fn from_buffer(buf: Box<[f32]>) -> Self {
        if buf.len() != W::size() * H::size() {
            panic!(
                "Invalid buffer size: expected {}, found {}",
                H::size() * W::size(),
                buf.len()
            );
        }

        Matrix(buf, PhantomData, PhantomData)
    }

    pub fn _transpose(&self) -> Matrix<H, W> {
        let mut new = Matrix::<H, W>::default();
        for i in 0..H::size() {
            for k in 0..W::size() {
                new[k][i] = self[i][k]
            }
        }

        new
    }

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::default();
        for i in 0..H::size() {
            for k in 0..W::size() {
                result[i][k] = self[i][k] + other[i][k]
            }
        }

        result
    }

    #[inline]
    pub fn dot<W2: SizeMarker>(&self, other: &Matrix<W2, W>) -> Matrix<W2, H> {
        let mut result = Matrix::<W2, H>::default();
        for i in 0..H::size() {
            for j in 0..W2::size() {
                for k in 0..W::size() {
                    result[i][j] += self[i][k] * other[k][j];
                }
            }
        }

        result
    }

    #[inline]
    pub fn relu(&self, alpha: f32) -> Self {
        let mut result = Self::default();
        for i in 0..H::size() {
            for j in 0..W::size() {
                if self[i][j] < 0.0 {
                    result[i][j] = self[i][j] * alpha;
                } else {
                    result[i][j] = self[i][j]
                }
            }
        }

        result
    }
}

impl<W: SizeMarker> Matrix<W, Size1> {
    pub fn from_array(arr: Box<[f32]>) -> Matrix<W, Size1> {
        if arr.len() != W::size() {
            panic!(
                "Invalid array size: expected {}, found {}",
                W::size(),
                arr.len()
            );
        }

        Matrix(arr, PhantomData, PhantomData)
    }
}

impl<W: SizeMarker, H: SizeMarker> Index<usize> for Matrix<W, H> {
    type Output = [f32];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * W::size();
        let end = start + W::size();
        &self.0[start..end]
    }
}

impl<W: SizeMarker, H: SizeMarker> IndexMut<usize> for Matrix<W, H> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = index * W::size();
        let end = start + W::size();
        &mut self.0[start..end]
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::{size::*, Matrix, SizeMarker};
    use crate::model_data::tests::get_test_model;
    use crate::model_data::tests::MARGIN;
    use float_cmp::{approx_eq, ApproxEq};

    impl<W: SizeMarker, H: SizeMarker> Matrix<W, H> {
        pub fn approx_eq(&self, other: &Self) -> bool {
            for i in 0..H::size() {
                for j in 0..W::size() {
                    if !self[i][j].approx_eq(other[i][j], MARGIN) {
                        return false;
                    }
                }
            }

            return true;
        }
    }

    #[test]
    fn test_transpose() {
        let model = get_test_model();
        let original = model.weights.l2_kernel;
        let transposed = original._transpose();
        for i in 0..4 {
            assert!(approx_eq!(f32, original[i][0], transposed[0][i]));
        }
    }

    #[test]
    fn test_dot() {
        let model = get_test_model();
        let a = model.weights.l2_kernel;
        let b = a._transpose();
        let result = a.dot(&b);
        let mut acc = 0.0;
        for i in 0..4 {
            acc += a[i][0] * a[i][0];
        }
        approx_eq!(f32, result[0][0], acc);

        let _test = model.weights.l0_kernel.dot(&model.weights.l1_kernel);
    }

    #[test]
    fn test_add() {
        let m1 = Matrix::<Size1, Size1>::from_array(vec![1.0f32].into_boxed_slice());
        let m2 = Matrix::<Size1, Size1>::from_array(vec![1.0f32].into_boxed_slice());
        let result = m1.add(&m2);
        assert!(approx_eq!(f32, 2.0f32, result[0][0]));
    }

    #[test]
    fn test_relu() {
        for alpha in [0.0f32, 0.1, 0.01].iter() {
            let m1 = Matrix::<Size2, Size1>::from_array(vec![1.0, -1.0].into_boxed_slice());
            let expected = Matrix::<Size2, Size1>::from_array(vec![1.0, -alpha].into_boxed_slice());
            let relu = m1.relu(*alpha);
            assert!(relu.approx_eq(&expected));
        }
    }
}
