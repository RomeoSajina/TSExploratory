# TSExploratory

Evaluate your univariate time series on range of s:
- Base models:
    - Persistent 
    - Seasonal Persistent 

- Stats models:
    - AR
    - MA
    - ARMA
    - ARIMA
    - SARIMAX
    - UnobservedComponents
    -Prophet

- Neural network models:
    - BidirectionalLSTM
    - CNNLSTM
    - GRU
    - LSTM
    - TimeDistributedCNNLSTM
    - CNN
    - MultiCNN
    - MLP
    - AutoencoderMLP
    - AutoencoderCNN
    - AutoencoderMultiCNN
    - AutoencoderMLPLSTM
    - AutoencoderMLPGRU
    - AutoencoderRandomDropoutMLP
    - RandomDropoutLSTM
    - AutoencoderRandomDropoutBidirectionalLSTM
    - AutoencoderRandomDropoutTimeDistributedCNNLSTM
    - AutoencoderRandomDropoutCNNLSTM
    - AutoencoderCNNLSTMTimeDistributed
    - RandomDropoutGRU
    - AutoencoderRandomDropoutCNN
    - AutoencoderRandomDropoutMultiCNN
    - AutoencoderRandomDropoutMLPLSTM
    - AutoencoderRandomDropoutMLPGRU
    - ResNetClassification
    - ResNetLSTM


## Get started


To get started on your dataset you need to implement method `load_ts` in class `DataFactory` where you need to return a `Config` object:

```python
@staticmethod
def load_ts(end_date=None, target_date=None, data=None, use_cache=True):

    end_date = datetime.date(2019, 1, 5)
    target_date = datetime.date(2019, 1, 7)

    elem_list = [
        {"X": datetime.date(2019, 1, 1), "y": 10},
        {"X": datetime.date(2019, 1, 2), "y": 9},
        {"X": datetime.date(2019, 1, 3), "y": 11},
        {"X": datetime.date(2019, 1, 4), "y": 14},
        {"X": datetime.date(2019, 1, 5), "y": 7},
        {"X": datetime.date(2019, 1, 6), "y": 12},
        {"X": datetime.date(2019, 1, 7), "y": 9},
    ]

    ts = pd.DataFrame(elem_list)

    ts = ts.set_index("X", drop=True)

    config = Config.build(ts, end_date, target_date)

    return config
```


Then u can use it as follows:

```python

config = DataFactory.load_ts()


# Adjust number of history steps and predictiong range with Metata. Example:
config.apply_metadata(Metadata.version_1())


# Create models

# Base models
model = PersistentModelWrapper(config)
model = SeasonalPersistentModelWrapper(config)

# Stats models
model = ARModelWrapper(config)
model = MAModelWrapper(config)
model = ARMAModelWrapper(config)
model = ARIMAModelWrapper(config)
model = SARIMAXModelWrapper(config)
model = UnobservedComponentsModelWrapper(config)

model = ProphetModelWrapper(config)

# NN models
model = BidirectionalLSTMModelWrapper(config)
model = CNNLSTMModelWrapper(config)
model = GRUModelWrapper(config)
model = LSTMModelWrapper(config)
model = TimeDistributedCNNLSTMModelWrapper(config)
model = CNNModelWrapper(config)
model = MultiCNNModelWrapper(config)
model = MLPModelWrapper(config)
model = AutoencoderMLPModelWrapper(config)
model = AutoencoderCNNModelWrapper(config)
model = AutoencoderMultiCNNModelWrapper(config)
model = AutoencoderMLPLSTMModelWrapper(config)
model = AutoencoderMLPGRUModelWrapper(config)
model = AutoencoderRandomDropoutMLPModelWrapper(config)
model = RandomDropoutLSTMModelWrapper(config)
model = AutoencoderRandomDropoutBidirectionalLSTMModelWrapper(config)
model = AutoencoderRandomDropoutTimeDistributedCNNLSTMModelWrapper(config)
model = AutoencoderRandomDropoutCNNLSTMModelWrapper(config)
model = AutoencoderCNNLSTMTimeDistributedModelWrapper(config)
model = RandomDropoutGRUModelWrapper(config)
model = AutoencoderRandomDropoutCNNModelWrapper(config)
model = AutoencoderRandomDropoutMultiCNNModelWrapper(config)
model = AutoencoderRandomDropoutMLPLSTMModelWrapper(config)
model = AutoencoderRandomDropoutMLPGRUModelWrapper(config)
model = ResNetClassificationModelWrapper(config)
model = ResNetLSTMModelWrapper(config)


# Fit model
model.fit()

# Predict
model.predict()
model.predict(days=100)


# Plot
model.plot_train()
#model.save_train_figure()

model.plot_predict()
model.plot_predict_multiple()

Plotly.ZOOM = -50
model.plot_predict(425)
#model.save_prediction_figure()

```


**Check `main.py` for more code examples**