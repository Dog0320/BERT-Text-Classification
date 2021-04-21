# BERT-Text-Classification
BERT-Text-Classification pytorch implementation

## Requirements
```
Python3
transformers==3.1.0
torch==1.6.1+
argparse==1.4.0
```

## Prepare

* Download ``google_model.bin`` from [here](https://drive.google.com/drive/folders/1i67mPV1i2P2IMNTks2PtPeZsDnA8SVQN?usp=sharing), and save it to the ``assets/`` directory.
* Download ``dataset`` from [here](https://drive.google.com/drive/folders/1i67mPV1i2P2IMNTks2PtPeZsDnA8SVQN?usp=sharing), and save it to the ``data/`` directory.

### Classification example

Run example on ATIS dataset.
```
python3 train.py --data_dir data/atis/ --model_path /assets/
```
