use crate::Error;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

pub type Size = (usize, usize);

#[derive(Serialize, Deserialize, Debug)]
pub struct Matrix(Vec<Vec<f32>>);

impl Deref for Matrix {
    type Target = Vec<Vec<f32>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Matrix {
    pub fn new(size: Size) -> Self {
        Matrix(vec![vec![0.0f32; size.1]; size.0])
    }

    pub fn _transpose(&self) -> Self {
        let size = self.size();
        let mut new = Self::new((size.1, size.0));
        for i in 0..size.0 {
            for k in 0..size.1 {
                new[k][i] = self[i][k]
            }
        }
        new
    }

    pub fn from_array(vec: Vec<f32>) -> Self {
        Matrix(vec![vec])
    }

    pub fn dot(&self, other: &Self) -> Result<Self, Error> {
        let size_self = self.size();
        let size_other = other.size();
        if size_self.1 != size_other.0 {
            return Err(Error::IncompatibleDotMatrix(size_self, size_other));
        }
        let size_result = (size_self.0, size_other.1);
        let mut result = Matrix::new(size_result);
        for i in 0..size_result.0 {
            for j in 0..size_result.1 {
                let mut acc = 0.0f32;
                for k in 0..size_self.1 {
                    acc += self[i][k] * other[k][j];
                }
                result[i][j] = acc;
            }
        }

        Ok(result)
    }

    pub fn add(&self, other: &[f32]) -> Result<Self, Error> {
        let size_self = self.size();
        let size_other = (1, other.len());
        if size_self != size_other {
            return Err(Error::IncompatibleAddMatrix(size_self, size_other));
        }
        let mut result = Matrix::new(size_self);
        for i in 0..size_self.1 {
            result[0][i] = self[0][i] + other[i];
        }
        Ok(result)
    }

    pub fn size(&self) -> Size {
        let a = self.len();
        let b = if a > 0 { self[0].len() } else { 0 };
        (a, b)
    }

    pub fn relu(&self, alpha: f32) -> Self {
        let size_self = self.size();
        let mut result = Self::new(size_self);
        for i in 0..size_self.0 {
            for j in 0..size_self.1 {
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

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use crate::model_data::tests::get_test_model;
    use crate::model_data::tests::MARGIN;
    use float_cmp::{approx_eq, ApproxEq};

    impl Matrix {
        pub fn approx_eq(&self, other: &Self) -> bool {
            let self_size = self.size();
            if self_size != other.size() {
                return false;
            }
            for i in 0..self_size.0 {
                for j in 0..self_size.1 {
                    if !self[i][j].approx_eq(other[i][j], MARGIN) {
                        return false;
                    }
                }
            }

            return true;
        }
    }

    #[test]
    fn test_size() {
        let model = get_test_model();
        assert_eq!((20, 4), model.weights.l0_kernel.size());
        assert_eq!((4, 4), model.weights.l1_kernel.size());
        assert_eq!((4, 1), model.weights.l2_kernel.size());
    }

    #[test]
    fn test_new() {
        let size = (1, 20);
        let m = Matrix::new(size);
        assert_eq!(m.size(), size);
    }

    #[test]
    fn test_transpose() {
        let model = get_test_model();
        assert_eq!((4, 20), model.weights.l0_kernel._transpose().size());
        let original = model.weights.l2_kernel;
        let transposed = original._transpose();
        assert_eq!((1, 4), transposed.size());
        for i in 0..4 {
            assert!(approx_eq!(f32, original[i][0], transposed[0][i]));
        }
    }

    #[test]
    fn test_dot() {
        let model = get_test_model();
        let a = model.weights.l2_kernel;
        let b = a._transpose();
        assert!(a.dot(&a).is_err());
        let result = a.dot(&b).unwrap();
        let mut acc = 0.0;
        for i in 0..4 {
            acc += a[i][0] * a[i][0];
        }
        approx_eq!(f32, result[0][0], acc);

        let test = model
            .weights
            .l0_kernel
            .dot(&model.weights.l1_kernel)
            .unwrap();
        assert_eq!(test.size(), (20, 4));
    }

    #[test]
    fn test_add() {
        let m1 = Matrix::from_array(vec![1.0f32]);
        let result = m1.add(&vec![1.0f32]).unwrap();
        assert!(approx_eq!(f32, 2.0f32, result[0][0]));
    }

    #[test]
    fn test_relu() {
        for alpha in [0.0f32, 0.1, 0.01].iter() {
            let m1 = Matrix::from_array(vec![1.0f32, -1.0]);
            let expected = Matrix::from_array(vec![1.0f32, -alpha]);
            let relu = m1.relu(*alpha);
            assert!(relu.approx_eq(&expected));
        }
    }
}
