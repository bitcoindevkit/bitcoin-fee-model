
# How to generate a new model

## Log data

Have a [bitcoin_logger](https://github.com/RCasatta/bitcoin_logger) instance running for a while. eg: 

```
target/release/bitcoin-logger --zmq-address tcp://192.168.1.167:28332 --save-path /mnt/ssd/bitcoin_log --rpc-address http://127.0.0.1:8332 --cookie-path /mnt/ssd/bitcoin/.cookie
```

## Build the dataset

Make a csv with `bitcoin-csv` binary in the bitcoin_logger. eg: 

```
./target/release/bitcoin-csv --dataset-file /mnt/bigssd/bitcoin/dataset --load-path /mnt/big/bitcoin_log/
```

## Create the models

Use tensorflow python program at https://colab.research.google.com/drive/1js7MCPkggQGvFXeMijPZy4G2cWparzlZ 

Create a virtualenv with needed requirements

```
source venv/bin/activate
python model.py
python model_with_hurry.pu
```

There are two models because one is done for hurry tx: confirming in 1 or 2 blocks, and the other model for tx confirming from 3 to 1008 blocks 

## Copy the model

Copy the resulting dirs, like `20210221-220251` into this repo, under `models` dir.

update `build.rs` pointing to the new dirs in `default_models` var.

update test `test_vector` poiting to the new dirs

## Test

`cargo test`