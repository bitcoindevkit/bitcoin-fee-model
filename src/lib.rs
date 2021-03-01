use std::collections::HashMap;

use chrono::{DateTime, Datelike, NaiveDateTime, Timelike, Utc};

use crate::fee_bucket::FeeBuckets;
use crate::matrix::{size::*, SizeMarker};
use crate::model_data::ModelData;

mod error;
mod fee_bucket;
mod matrix;
mod model_data;

#[cfg(feature = "use-bitcoin")]
pub mod process_blocks;

#[cfg(feature = "use-bitcoin")]
pub extern crate bitcoin;

#[cfg(feature = "use-bitcoin")]
pub use process_blocks::process_blocks;

pub use error::Error;
pub use model_data::models::*;

pub struct FeeModel<N> {
    low: ModelData<Size20, N, Size1>,
    high: ModelData<Size20, N, Size1>,
}

impl<N: SizeMarker> FeeModel<N> {
    pub fn new(low: ModelData<Size20, N, Size1>, high: ModelData<Size20, N, Size1>) -> FeeModel<N> {
        FeeModel { low, high }
    }

    pub fn estimate_with_buckets(
        &self,
        block_target: u16,
        timestamp: Option<u32>,
        fee_buckets: &[u64],
        last_block_ts: u32,
    ) -> Result<f32, Error> {
        let mut input = HashMap::new();
        input.insert("confirms_in".to_string(), block_target as f32);

        let utc: DateTime<Utc> = match timestamp {
            Some(timestamp) => {
                let naive = NaiveDateTime::from_timestamp(timestamp as i64, 0);
                DateTime::from_utc(naive, Utc)
            }
            None => Utc::now(),
        };
        let day_of_week = utc.weekday().num_days_from_monday() as f32;
        input.insert("day_of_week".to_string(), day_of_week);
        input.insert("hour".to_string(), utc.hour() as f32);

        let delta = utc.timestamp() - last_block_ts as i64;
        input.insert("delta_last".to_string(), delta as f32);

        for i in 0..=15 {
            input.insert(format!("b{}", i), fee_buckets[i] as f32);
        }

        if block_target <= 2 {
            self.low.norm_predict(&input)
        } else {
            self.high.norm_predict(&input)
        }
    }

    /// compute the fee estimation given the desired `block_target`
    /// `timestamp` if None it's initialized to current time.
    /// `fee_rates` contains the fee rates of transactions in the last 10 blocks, only for transactions
    /// having inputs in this last 10 blocks (so the fee rate is known)
    /// `last_block_ts` last
    pub fn estimate(
        &self,
        block_target: u16,
        timestamp: Option<u32>,
        fee_rates: &[f64],
        last_block_ts: u32,
    ) -> Result<f32, Error> {
        let fee_buckets = FeeBuckets::new(50, 500.0).get(fee_rates);
        self.estimate_with_buckets(block_target, timestamp, &fee_buckets, last_block_ts)
    }
}

#[cfg(test)]
mod tests {
    use crate::model_data::tests::BUCKETS;
    use crate::FeeModel;
    use crate::{get_model_high, get_model_low};

    #[test]
    pub fn test_estimate() {
        let model = FeeModel::new(get_model_low(), get_model_high());
        let ts = 1613708045u32;
        let one = model
            .estimate_with_buckets(1, Some(ts), &BUCKETS, ts - 300)
            .unwrap();
        let two = model
            .estimate_with_buckets(2, Some(ts), &BUCKETS, ts - 300)
            .unwrap();
        assert!(one > two, "1 block ({}) > 2 ({})", one, two);
    }
}
