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

* Download ``google_model.bin`` from [here](https://drive.google.com/drive/folders/11i463eaaVvBrulLzSmUHdjFRgO_txBnU?usp=sharing), and save it to the ``assets/`` directory.
* Download ``dataset`` from [here](https://drive.google.com/drive/folders/1LyctA87Ve2oNb7nxovFzZbT_td9CCV24?usp=sharing), and save it to the ``data/`` directory.

### Model Training

Run example on ATIS dataset.
```
python3 train.py --data_dir data/atis/ --model_path /assets/
```
#### To use your own dataset,  modify the DataProcessor in ``data_utils.py``.

### Model Evaluation

Run example on ATIS dataset.
```
python3 evaluate.py --data_dir data/atis/ --model_path /assets/
```

### Model Prediction

Run example on ATIS dataset.
```
python3 predict.py --data_dir data/atis/ --model_path /assets/
```

