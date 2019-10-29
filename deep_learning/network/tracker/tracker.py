from .visualizer.visualizer import Visualizer
import numpy as np

class Tracker:

    def __init__(self, network):

        self._visualizer = Visualizer()
        self._network = network

        self._train_cost = []
        self._valid_cost = []
        self._train_metric = []
        self._valid_metric = []
        self._epochs = []
        self._epoch = 0 

        self._epoch_train_cost = 0.0
        self._epoch_valid_cost = 0.0
        self._epoch_train_metric = 0.0
        self._epoch_valid_metric = 0.0
        self._epoch_batches = 0

        self._processed_samples = 0
        self._tracking_rate = 1
        self._showing_rate = 1 

        self._x_train = None
        self._x_valid = None
        self._x_valid = None
        self._y_valid = None



    @property
    def costs(self):
        return self._cost



    @property
    def metrics(self):
        return self._metric



    def set_tracking_rate(self, rate):
        self._tracking_rate = rate



    def set_showing_rate(self, rate):
        self._showing_rate = rate


    
    def set_training_set(self, x_train, y_train):
        self._x_train = x_train
        self._y_train = y_train



    def set_validation_set(self, x_valid, y_valid):
        self._x_valid = x_valid
        self._y_valid = y_valid



    def start_new_epoch(self, epoch, num_epochs):
        self._epoch_train_cost = 0.0
        self._epoch_train_metric = 0.0
        self._processed_samples = 0
        self._epoch_batches = 0
        self._epoch += 1

        self._epoch_valid_cost = None
        self._epoch_valid_metric = None

        if self.time_to_show():
            total_epochs = (self._epoch
                         + (num_epochs - epoch - 1))
            self._visualizer.show_epoch_header(
                    self._epoch, total_epochs)



    def time_to_show(self):
        return (self._epoch-1) % self._showing_rate == 0



    def track(self, x_batch, y_batch, y_pred):
        self._epoch_train_cost += (
                self._network.compute_cost(y_batch, y_pred))

        self._epoch_train_metric += (
                self._network.compute_metric(y_batch, y_pred))

        self._processed_samples += y_batch.shape[-1]
        self._epoch_batches += 1

        if self.time_to_show():
            total_samples = self._x_train.shape[-1]
            self._visualizer.show_epoch_progress(
                    self._processed_samples, total_samples)



    def finish_current_epoch(self):
        self._epoch_train_cost /= self._epoch_batches
        self._epoch_train_metric /= self._epoch_batches
        
        if self._x_valid is not None and self._y_valid is not None:
            y_pred = self._network.predict(self._x_valid)
            self._epoch_valid_cost = (
                    self._network.compute_cost(self._y_valid, y_pred))
            self._epoch_valid_metric = (
                    self._network.compute_metric(self._y_valid, y_pred))

        else:
            self._epoch_valid_cost = self.undefined_array(
                    self._epoch_train_cost.shape)
            self._epoch_valid_metric = self.undefined_array(
                    self._epoch_train_metric.shape)

        if self.time_to_track():
            self._train_cost.append(np.reshape(self._epoch_train_cost, -1))
            self._valid_cost.append(np.reshape(self._epoch_valid_cost, -1))
            self._train_metric.append(np.reshape(self._epoch_train_metric, -1))
            self._valid_metric.append(np.reshape(self._epoch_valid_metric, -1))
            self._epochs.append(self._epoch)

        if self.time_to_show():
            self._visualizer.show_epoch_tracking(
                    self._epoch_train_cost,
                    self._epoch_valid_cost,
                    self._epoch_train_metric,
                    self._epoch_valid_metric)



    def undefined_array(self, shape):
        return np.full(shape, None)



    def time_to_track(self):
        return (self._epoch-1) % self._tracking_rate == 0



    def show_tracking(self, epoch=None):
        if epoch is None:
            epoch = self._epoch
            train_cost = self._epoch_train_cost
            valid_cost = self._epoch_valid_cost
            train_metric = self._epoch_train_metric
            valid_metric = self._epoch_valid_metric

        else:
            index = self.epoch_index(epoch)
            epoch = self._epochs[index]

            train_cost = self._train_cost[index]
            valid_cost = self._valid_cost[index]

            train_metric = self._train_metric[index]
            valid_metric = self._valid_metric[index]

        self._visualizer.show_epoch_header(epoch, self._epoch)
        self._visualizer.show_epoch_tracking(
                train_cost, valid_cost, train_metric, valid_metric)



    def epoch_index(self, epoch):
        return np.searchsorted(self._epochs, epoch)


    
    def plot_tracking(self):
        return self._visualizer.plot_tracking(
                self._epochs, self._train_cost, self._valid_cost,
                self._train_metric, self._valid_metric)



    def delete_until(self, epoch):
        index = self.epoch_index(epoch)
        self._epochs = self._epochs[index:]
        self._train_cost = self._train_cost[index:]
        self._valid_cost = self._valid_cost[index:]
        self._train_metric = self._train_metric[index:]
        self._valid_metric = self._valid_metric[index:]



    def reset_tracking(self):

        self._visualizer.reset()

        self._train_cost = []
        self._valid_cost = []
        self._train_metric = []
        self._valid_metric = []
        self._epochs = []
        self._epoch = 0

        self._epoch_cost = 0.0
        self._epoch_metric = 0.0
        self._epoch_batches = 0
        self._processed_samples = 0

        self._x_train = None
        self._x_valid = None
        self._x_valid = None
        self._y_valid = None


