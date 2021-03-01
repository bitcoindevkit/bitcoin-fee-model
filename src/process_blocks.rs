use crate::Error;
use bitcoin::{Block, Transaction, Txid};
use std::collections::HashMap;

pub struct Transactions {
    txs: HashMap<Txid, Transaction>, // TODO use &Transaction to avoid clones
    txs_output_values: HashMap<Txid, OutputValues>,
    last_block_ts: u32,
}

pub type OutputValues = Box<[u64]>;

pub fn process_blocks(blocks: &[bitcoin::Block; 10]) -> Result<(Vec<f64>, u32), Error> {
    let txs = Transactions::from_blocks(blocks)?;
    let fee_rates = txs.fee_rates();
    let last_block_ts = txs.last_block_ts();
    Ok((fee_rates, last_block_ts))
}

impl Transactions {
    pub fn from_blocks(blocks: &[Block; 10]) -> Result<Self, Error> {
        let mut prev = blocks[0].header.block_hash();
        for block in blocks.iter().skip(1) {
            if prev != block.header.prev_blockhash {
                return Err(Error::UnconnectedBlocks);
            }
            prev = block.block_hash();
        }
        let mut txs: HashMap<Txid, Transaction> = HashMap::new();
        let mut time = None;
        for block in blocks {
            if block.txdata.len() > 1 && time.is_none() {
                time = Some(block.header.time);
            }
            for tx in block.txdata.iter() {
                txs.insert(tx.txid(), tx.clone());
            }
        }

        Ok(Self::from_txs(
            txs,
            time.ok_or_else(|| Error::LastTsMissing)?,
        ))
    }
    pub fn from_txs(txs: HashMap<Txid, Transaction>, last_block_ts: u32) -> Self {
        let mut txs_output_values: HashMap<Txid, OutputValues> = HashMap::new();
        for (txid, tx) in txs.iter() {
            let output_values: Vec<_> = tx.output.iter().map(|e| e.value).collect();
            txs_output_values.insert(*txid, output_values.into_boxed_slice());
        }
        Transactions {
            txs,
            txs_output_values,
            last_block_ts,
        }
    }

    // fee rate in sat/vbytes
    pub fn fee_rate(&self, txid: &Txid) -> Option<f64> {
        let tx = self.txs.get(txid)?;
        let fee = self.absolute_fee(tx)?;
        Some((fee as f64) / (tx.get_weight() as f64 / 4.0))
    }

    pub fn fee_rates(&self) -> Vec<f64> {
        self.txs.keys().filter_map(|tx| self.fee_rate(tx)).collect()
    }

    fn absolute_fee(&self, tx: &Transaction) -> Option<u64> {
        let sum_outputs: u64 = tx.output.iter().map(|o| o.value).sum();
        let mut sum_inputs: u64 = 0;
        for input in tx.input.iter() {
            let outputs_values = self.txs_output_values.get(&input.previous_output.txid)?;
            sum_inputs += outputs_values[input.previous_output.vout as usize];
        }
        Some(sum_inputs - sum_outputs)
    }

    pub fn last_block_ts(&self) -> u32 {
        self.last_block_ts
    }
}

#[cfg(test)]
mod tests {
    use super::process_blocks;
    use crate::Error;
    use bitcoin::blockdata::constants::genesis_block;
    use bitcoin::{Block, Network};
    use std::convert::TryInto;

    #[test]
    fn test_blocks() {
        let block = genesis_block(Network::Bitcoin);

        let mut blocks: [Block; 10] = vec![block; 10].try_into().unwrap();
        let err = process_blocks(&blocks).unwrap_err();
        assert!(matches!(err, Error::UnconnectedBlocks));

        // make them connected
        let mut current_hash = blocks[0].header.block_hash();
        for block in blocks.iter_mut().skip(1) {
            block.header.prev_blockhash = current_hash;
            current_hash = block.block_hash();
        }
        let err = process_blocks(&blocks).unwrap_err();
        assert!(matches!(err, Error::LastTsMissing));

        // make a fake non empty (more than 1 tx) block
        let tx = blocks[0].txdata[0].clone();
        blocks[0].txdata.push(tx);
        process_blocks(&blocks).unwrap();
    }
}
