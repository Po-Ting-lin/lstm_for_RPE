"""
make sure sys path includes RPEImageProcess
"""
import numpy as np

from lstm.model_gen import bulid_model, get_lr_metric
from lstm.data_gen import data_generator
from lstm.callbacks_gen import getCallback
from lstm.evaluate_result import evaluate_model_performance, evaluation, show_final_history
from lstm.visualize_result import plot_time_series_data