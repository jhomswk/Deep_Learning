import matplotlib.pyplot as plt
import numpy as np

class Visualizer:


    def __init__(self):
        self._prefix = self._suffix = " "
        self._separator = "|"
        self._undefined = "---"
        self._precision = 3 
        self._bar_width = 25 
        self._status_width = 0
        self._cell_width = self.cell_width()
        self._header_width = self.header_width() 

        plt.rc("font", family="serif", weight="light", size=10)



    def cell_width(self):
        width =  1 # space
        width += 1 # digit
        width += 1 # dot
        width += self._precision
        width += 4 # exponent
        width += 1 # space
        return max(width, 14)
    


    def header_width(self):
        width = 1 # space
        width += len(self._prefix)
        width += self._cell_width
        width += len(self._separator)
        width += self._cell_width
        width += len(self._suffix)
        width += 1 # space
        return width



    def show_epoch_header(self, current_epoch, total_epochs):
        header = f"epoch {current_epoch}/{total_epochs}"
        header = f"\n{header:^{self._header_width}}"
        header = f"{header}\n {'-'*(self._header_width-2)} \n"
        print(header)



    def show_epoch_progress(self, completed, total):
        self.clear_status()
        self.show_status(completed, total)



    def clear_status(self):
        print(" "*self._status_width, end="\r")



    def show_status(self, completed, total):
        status = self.build_status_bar(completed, total)
        status = f"processing : {completed}/{total} {status}"
        self._status_width = len(status) + 1
        print(status, end="\r")



    def build_status_bar(self, completed, total):
        progress = self._bar_width*completed//total
        remaining = self._bar_width - progress
        status = f"[{progress*'='}{remaining*'.'}]"
        return status



    def show_epoch_tracking(self,
            train_cost, valid_cost, train_metric, valid_metric):

        self.clear_status()

        self.build_table(
                np.reshape(train_cost, -1),
                np.reshape(valid_cost, -1),
                "train cost", "valid cost")

        self.build_table(
                np.reshape(train_metric, -1),
                np.reshape(valid_metric, -1),
                "train metric", "valid metric")

        self._status_width = 0



    def build_table(self, train, valid, train_header, valid_header):

        dashes = '-'*(self._cell_width-2)

        if valid[0] is None:
            valid = np.full(valid.shape, self._undefined)

        print(self.build_row((train_header, valid_header)))
        print(self.build_row((dashes, dashes)))
        print("\n".join(map(self.build_row, zip(train, valid))))
        print("")



    def center(self, value):
        return f"{value:^{self._cell_width}}"



    def format(self, value):
        return self.center(f"{value:.0{self._precision}e}"
                if isinstance(value, (float, int)) else f"{value}")



    def build_row(self, content):
        row = self._separator.join(map(self.format, content))
        row = f"{self._prefix}{row}{self._suffix}"
        row = f"{row:^{self._header_width}}"
        return row



    def plot_tracking(self,
            epochs, train_cost, valid_cost, train_metric, valid_metric):

        figure = plt.figure(figsize=(10, 10))

        self._plot_tracking(
                epochs, train_cost, valid_cost, "Cost", figure, 0)

        self._plot_tracking(
                epochs, train_metric, valid_metric, "Metric", figure, 1)

        figure.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.show()



    def _plot_tracking(self, epochs, train, valid, label, figure, row):

        unidimensional = train[-1].shape[0] == 1

        if unidimensional:
            self.plot_train_valid_together(
                    epochs, train, valid, label, figure, row)
        else:
            self.plot_train_valid_separate(
                    epochs, train, valid, label, figure, row)
        


    def plot_train_valid_separate(self,
            epochs, train, valid, label, figure, row):
        
        ax1 = figure.add_subplot(2, 2, 2*row+1)
        ax1.plot(epochs, train)
        ax1.set_ylabel(rf"{label}", size=10, labelpad=15)
        ax1.set_title(r"Training", size=10, pad=15)
        ax1.set_xlabel(r"Epoch", size=10, labelpad=15)
        ax1.grid(linestyle="--")

        ax2 = figure.add_subplot(2, 2, 2*(row+1), sharex=ax1, sharey=ax1)
        ax2.plot(epochs, valid)
        ax2.set_title(r"Validation", size=10, pad=15)
        ax2.set_xlabel(r"Epoch", size=10, labelpad=15)
        ax2.grid(linestyle="--")
            


    def plot_train_valid_together(self,
            epochs, train, valid, label, figure, row):

        ax = figure.add_subplot(2, 1, row+1)
        ax.plot(epochs, train, label=r"training")
        ax.plot(epochs, valid, label=r"validation")
        ax.set_ylabel(rf"{label}", size=10, labelpad=15)
        ax.set_xlabel(r"Epoch", size=10, labelpad=15)
        ax.grid(linestyle="--")
        ax.legend()



    def reset(self):
        self._status_width = 0
    

