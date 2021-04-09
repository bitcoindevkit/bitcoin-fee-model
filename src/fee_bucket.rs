#[derive(Debug)]
pub struct FeeBuckets {
    buckets_limits: Vec<f64>,
}

impl FeeBuckets {
    pub fn new(increment_percent: u32, upper_limit: f64) -> Self {
        let buckets_limits = create_buckets_limits(increment_percent, upper_limit);
        FeeBuckets { buckets_limits }
    }

    pub fn get(&self, rates: &[f64]) -> Vec<u64> {
        let mut buckets = vec![0u64; self.buckets_limits.len()];
        for rate in rates {
            let index = self
                .buckets_limits
                .iter()
                .position(|e| e > rate)
                .unwrap_or(self.buckets_limits.len() - 1);
            buckets[index] += 1;
        }
        buckets
    }
}

pub fn create_buckets_limits(increment_percent: u32, upper_limit: f64) -> Vec<f64> {
    let mut buckets_limits = vec![];
    let increment_percent = 1.0f64 + (increment_percent as f64 / 100.0f64);
    let mut current_value = 1.0f64;
    loop {
        if current_value >= upper_limit {
            break;
        }
        current_value *= increment_percent;
        buckets_limits.push(current_value);
    }
    buckets_limits
}
