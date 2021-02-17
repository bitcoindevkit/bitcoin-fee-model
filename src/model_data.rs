use crate::matrix::Matrix;
use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufReader, Read};

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelData {
    pub norm: FieldsDescribe,
    pub weights: Weights,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Weights {
    #[serde(rename = "dense/bias:0")]
    pub l0_bias: Vec<f32>,
    #[serde(rename = "dense/kernel:0")]
    pub l0_kernel: Matrix,

    #[serde(rename = "dense_1/bias:0")]
    pub l1_bias: Vec<f32>,
    #[serde(rename = "dense_1/kernel:0")]
    pub l1_kernel: Matrix,

    #[serde(rename = "dense_2/bias:0")]
    pub l2_bias: Vec<f32>,
    #[serde(rename = "dense_2/kernel:0")]
    pub l2_kernel: Matrix,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FieldsDescribe {
    mean: HashMap<String, f32>,
    std: HashMap<String, f32>,
    fields: Vec<String>,
}

impl ModelData {
    pub fn from_reader<R: Read>(reader: R) -> Result<Self, Error> {
        let buffer = BufReader::new(reader);
        let model: Self = serde_cbor::from_reader(buffer)?;
        Ok(model)
    }

    pub fn predict(&self, input: &Matrix) -> Result<f32, Error> {
        let a1 = input.dot(&self.weights.l0_kernel)?;
        let a2 = a1.add(&self.weights.l0_bias)?;
        let a3 = a2.relu();

        let b1 = a3.dot(&self.weights.l1_kernel)?;
        let b2 = b1.add(&self.weights.l1_bias)?;
        let b3 = b2.relu();

        let c1 = b3.dot(&self.weights.l2_kernel)?;
        let c2 = c1.add(&self.weights.l2_bias)?;

        // fee estimation should not go under 1.0, however, model could go under 1.0 in some rare
        // case, preventing this in the model activation function cause the model to get stuck in
        // some cases, thus the model gives penalties to values under 1.0 but does not guarantee
        // absolute absence of values less than 1.0, thus we need to enforce here.
        Ok(c2[0][0].max(1.0))
    }

    pub fn norm(&self, input: &HashMap<String, f32>) -> Result<Matrix, Error> {
        let mut result = vec![];
        for field in self.norm.fields.iter() {
            let x = input.get(field).unwrap_or(&0.0);
            let std = self
                .norm
                .std
                .get(field)
                .ok_or_else(|| Error::MissingStdData(field.clone()))?;
            let mean = self
                .norm
                .mean
                .get(field)
                .ok_or_else(|| Error::MissingMeanData(field.clone()))?;
            let res = (x - mean) / std;
            result.push(res)
        }
        Ok(Matrix::from_array(result))
    }

    pub fn norm_predict(&self, input: &HashMap<String, f32>) -> Result<f32, Error> {
        let input = self.norm(input)?;
        self.predict(&input)
    }
}

#[cfg(test)]
pub mod tests {
    use crate::matrix::Matrix;
    use crate::ModelData;
    use float_cmp::{ApproxEq, F32Margin};
    use std::collections::HashMap;
    use std::io::Cursor;

    pub const MARGIN: F32Margin = F32Margin {
        ulps: 2,
        epsilon: 0.0001,
    };

    pub fn get_test_model() -> ModelData {
        let model_bytes = include_bytes!("../models/test_model.cbor");
        assert_eq!(1784, model_bytes.len(), "test model bytes not expected");
        let model = ModelData::from_reader(Cursor::new(model_bytes));
        assert!(model.is_ok(), "can't restore model from bytes");
        model.unwrap()
    }

    #[rustfmt::skip]
    pub fn get_test_input() -> Matrix {
        Matrix::from_array(vec![-0.27729391, -0.15976526, -0.28554924, -0.3014742 , -0.38580752,
                                -0.40276291, -0.52907508, -0.5511707 , -0.72837495, -0.39697618,
                                0.10225281,  3.66144889, -0.34918782, -0.72627474, -0.69620141,
                                -0.56535236, -0.77867051, -0.39794625, -1.41245707,  0.98371395])
    }

    pub const BUCKETS: [u64; 16] = [
        2u64, 0, 0, 2, 5, 6, 14, 20, 95, 394, 4449, 1954, 282, 193, 33, 19,
    ];

    pub fn get_test_pre_norm() -> HashMap<String, f32> {
        let mut map = HashMap::new();
        map.insert("confirms_in".to_string(), 2.0);
        for (i, el) in BUCKETS.iter().enumerate() {
            map.insert(format!("b{}", i), *el as f32);
        }
        map.insert("delta_last".to_string(), 422.0);
        map.insert("day_of_week".to_string(), 0.0);
        map.insert("hour".to_string(), 19.0);
        map
    }

    pub fn get_test_result() -> f32 {
        62.726908
    }

    #[test]
    fn test_predict() {
        let model = get_test_model();
        let input = get_test_input();
        assert!(get_test_result().approx_eq(model.predict(&input).unwrap(), MARGIN))
    }

    #[test]
    fn test_vector() {
        let model = get_test_model();
        let input = get_test_input();
        assert_eq!((1, 20), input.size());

        let a1 = input.dot(&model.weights.l0_kernel).unwrap();
        let a1_expected = Matrix::from_array(vec![3.16310973, 2.48554417, 5.19779316, 4.20878246]);
        assert!(a1.approx_eq(&a1_expected));
        let a2 = a1.add(&model.weights.l0_bias).unwrap();
        let a2_expected = Matrix::from_array(vec![2.4573005, -0.93953398, 7.56838456, 0.01959875]);
        assert!(a2.approx_eq(&a2_expected));
        let a3 = a2.relu();

        let b1 = a3.dot(&model.weights.l1_kernel).unwrap();
        let b2 = b1.add(&model.weights.l1_bias).unwrap();
        let b3 = b2.relu();
        let b3_expected = Matrix::from_array(vec![13.76071502, 0., 0., 0.]);
        assert!(b3.approx_eq(&b3_expected));

        let c1 = b3.dot(&model.weights.l2_kernel).unwrap();
        let c2 = c1.add(&model.weights.l2_bias).unwrap();

        assert!(get_test_result().approx_eq(c2[0][0], MARGIN))
    }

    #[test]
    #[rustfmt::skip]
    fn test_norm() {
        let model = get_test_model();
        let expected = get_test_input();
        let norm = model.norm(&get_test_pre_norm()).unwrap();
        assert!(norm.approx_eq(&expected), "normalization is wrong");
    }
}
