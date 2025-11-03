# rps_plotter.py
import matplotlib.pyplot as plt

class RPSPlotter:
    def __init__(self, ai):
        self.ai = ai
        self.fig, (self.ax_weights, self.ax_conf) = plt.subplots(2, 1, figsize=(8, 6))
        plt.ion()
        plt.show()

    def update_plot(self):
        # Clear previous plots
        self.ax_weights.cla()
        self.ax_conf.cla()

        # Plot strategy weights
        names = self.ai.strategy_names
        weights = self.ai.weights
        self.ax_weights.bar(names, weights, color='dodgerblue')
        self.ax_weights.set_ylim(0, max(1.5, max(weights)*1.1))
        self.ax_weights.set_title("Strategy Weights (EXP3)")
        self.ax_weights.set_ylabel("Weight")

        # Plot prediction confidence
        self.ax_conf.bar(['Confidence'], [self.ai.prediction_confidence], color='orange')
        self.ax_conf.set_ylim(0, 1)
        self.ax_conf.set_title("Prediction Confidence")
        self.ax_conf.set_ylabel("Confidence")

        plt.tight_layout()
        plt.pause(0.001)
