use bitcoin_fee_model::FeeModel;
use bitcoin_fee_model::{get_model_high, get_model_low};

pub const BUCKETS: [u64; 16] = [
    13u64, 1, 32, 24, 14, 62, 1174, 453, 197, 291, 333, 3304, 307, 229, 36, 58,
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = FeeModel::new(get_model_low(), get_model_high());

    let ts = 1613708045u32;
    let one = model.estimate_with_buckets(1, Some(ts), &BUCKETS, ts - 300)?;
    let two = model.estimate_with_buckets(2, Some(ts), &BUCKETS, ts - 300)?;

    dbg!(&one);
    dbg!(&two);

    Ok(())
}
