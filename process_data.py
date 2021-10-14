import numpy as np
import random
import math
import time
import pandas as pd
import json
from os import listdir
from seq_cl import seq_cl
from traditional_baseline import tradition_b


class read_data_covid():
    """
    Loading data, mean and std are pre-computed
    """

    def __init__(self,data,label):
        """
        data: Input data, must be nxm, where n is time dimension, m is feature dimension
        label: the corresponding label associated with data
        time_sequence: how long is the time steps
        prediction_window: window size for event prediction period
        """
        self.data = data
        self.label = label
        self.feature_length = self.data.shape[1]
        self.time_sequence = 6
        self.predict_window = 0

    def return_tensor_data_dynamic(self, label, hr_onset, patient_table):
        """
        label: single patient label
        hr_onset: event happening time(death or live)
        patient_table: the single nxm patient feature table
        """
        self.one_data_tensor = np.zeros((self.time_sequence, self.feature_length))
        if label == 1:
            self.logit_label = 1
        else:
            self.logit_label = 0

        self.predict_window_start = hr_onset - self.predict_window

        self.one_data_tensor = self.assign_value(self.predict_window_start, patient_table)

    def assign_value(self, hr_back, patient_table):
        one_data_sample = np.zeros((self.time_sequence, self.vital_length))
        for i in range(self.time_sequence):
            self.hr_current = np.float(hr_back - self.time_sequence + i)
            if self.hr_current < 0:
                self.hr_current = 0.0
            for j in self.feature_length:
                one_data_sample[i, j] = patient_table[self.hr_current,j]


        return one_vital_sample



