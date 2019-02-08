import numpy as np
import matplotlib.pyplot as plt

# Show a 2D plot with the data in beat
def display_signal(beat):
    plt.plot(beat)
    plt.ylabel('Signal')
    plt.show()

# Class for RR intervals features
class RR_intervals:
    def __init__(self):
        # Instance atributes
        # self.pre_R = np.array([])
        self.pre_R = np.array([])
        self.post_R = np.array([])
        self.local_R = np.array([])
        self.global_R = np.array([])

class mit_db:
    def __init__(self):
        # Instance attributes
        self.filename = []
        self.raw_signal = []
        self.beat = np.empty([])  # record, beat, lead
        self.class_ID = []
        self.valid_R = []
        self.R_pos = []
        self.orig_R_pos = []
