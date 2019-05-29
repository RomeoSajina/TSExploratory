from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.platform import tf_logging as logging
import numpy as np


class EarlyStoppingAtMinLoss(Callback):
        """Stop training when a monitored quantity has stopped improving.
           Szop training when loss is NaN

        Arguments:
            monitor: Quantity to be monitored.
            min_delta: Minimum change in the monitored quantity
                to qualify as an improvement, i.e. an absolute
                change of less than min_delta, will count as no
                improvement.
            patience: Number of epochs with no improvement
                after which training will be stopped.
            verbose: verbosity mode.
            mode: One of `{"auto", "min", "max"}`. In `min` mode,
                training will stop when the quantity
                monitored has stopped decreasing; in `max`
                mode it will stop when the quantity
                monitored has stopped increasing; in `auto`
                mode, the direction is automatically inferred
                from the name of the monitored quantity.
            baseline: Baseline value for the monitored quantity.
                Training will stop if the model doesn't show improvement over the
                baseline.
            restore_best_weights: Whether to restore model weights from
                the epoch with the best value of the monitored quantity.
                If False, the model weights obtained at the last step of
                training are used.
            margin_loss: Margin loss that must be satisfied before stopping
        """

        def __init__(self,
                     monitor='val_loss',
                     min_delta=0,
                     patience=0,
                     verbose=0,
                     mode='auto',
                     baseline=None,
                     restore_best_weights=False,
                     margin_loss=None,
                     terminate_on_nan=True):
            super(EarlyStoppingAtMinLoss, self).__init__()

            self.monitor = monitor
            self.patience = patience
            self.verbose = verbose
            self.baseline = baseline
            self.min_delta = abs(min_delta)
            self.wait = 0
            self.stopped_epoch = 0
            self.restore_best_weights = restore_best_weights
            self.best_weights = None
            self.last_weights = None
            self.margin_loss = margin_loss
            self.terminate_on_nan = terminate_on_nan
            self.last_epoch = 0

            if mode not in ['auto', 'min', 'max']:
                logging.warning('EarlyStopping mode %s is unknown, '
                                'fallback to auto mode.', mode)
                mode = 'auto'

            if mode == 'min':
                self.monitor_op = np.less
            elif mode == 'max':
                self.monitor_op = np.greater
            else:
                if 'acc' in self.monitor:
                    self.monitor_op = np.greater
                else:
                    self.monitor_op = np.less

            if self.monitor_op == np.greater:
                self.min_delta *= 1
            else:
                self.min_delta *= -1

        def on_train_begin(self, logs=None):
            # Allow instances to be re-used
            self.wait = 0
            self.stopped_epoch = 0
            if self.baseline is not None:
                self.best = self.baseline
            else:
                self.best = np.Inf if self.monitor_op == np.less else -np.Inf

        def on_epoch_end(self, epoch, logs=None):

            self.last_epoch = epoch

            current = self.get_monitor_value(logs)
            if current is None:
                return

            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0

                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()

                self.last_weights = self.model.get_weights()

            else:
                self.wait += 1
                if self.wait >= self.patience:
                    # Added to filter only if loss is less than margin_loss
                    if self.monitor_op(current, self.margin_loss):
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                        if self.restore_best_weights:
                            if self.verbose > 0:
                                print('Restoring model weights from the end of the best epoch.')
                            self.model.set_weights(self.best_weights)
                    else:
                        self.wait = self.patience/2 if self.patience > 1 else 0

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0 and self.verbose > 0:
                print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

        def get_monitor_value(self, logs):
            logs = logs or {}
            monitor_value = logs.get(self.monitor)
            if monitor_value is None:
                logging.warning('Early stopping conditioned on metric `%s` '
                                'which is not available. Available metrics are: %s',
                                self.monitor, ','.join(list(logs.keys())))
            return monitor_value

        def on_batch_end(self, batch, logs=None):
            logs = logs or {}
            loss = logs.get('loss')

            if loss is not None and self.terminate_on_nan:

                if np.isnan(loss) or np.isinf(loss):
                    print('Batch %d: Invalid loss, terminating training' % (batch))

                    self.model.stop_training = True

                    if self.restore_best_weights:
                        if self.verbose > 0:
                            print('Restoring model weights from the end of the best epoch.')
                        self.model.set_weights(self.best_weights)

                    elif self.last_weights is not None:
                        if self.verbose > 0:
                            print('Restoring model weights from the end of the last epoch.')
                        self.model.set_weights(self.last_weights)


"""
class RestoreAndTerminateOnNaN(Callback):
    
    def __init__(self):
        self.best_weights = None
        super(RestoreAndTerminateOnNaN, self).__init__()
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        
        if loss is not None:
        
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
            
                if self.best_weights is not None:
                    print('Restoring model weights from the end of the last epoch.')
                    self.model.set_weights(self.best_weights)
    
                self.model.stop_training = True
        
        self.best_weights = self.model.get_weights()     
"""
