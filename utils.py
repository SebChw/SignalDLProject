from tensorflow.keras.layers import LSTM, GRU, Dense, Input, RepeatVector, TimeDistributed, SimpleRNN
from tensorflow.keras.layers import Reshape, GlobalMaxPool1D, Lambda, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import tensorflow.keras.backend as K
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def get_callbacks(folder_to_save_weights : str):
    Path(folder_to_save_weights).mkdir(exist_ok=True)

    early = EarlyStopping(patience=7, restore_best_weights=True)
    reduce = ReduceLROnPlateau(monitor='loss', patience=6)
    model_checkpoint = ModelCheckpoint(filepath= folder_to_save_weights + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                        save_weights_only=True, monitor='val_binary_accuracy', 
                                        mode='max', save_best_only=True)
    
    return [early, reduce, model_checkpoint]

def smooth(x,window_len=11,window='hanning'):
    """Function taken from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html"""
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2):-(window_len//2-1)]

def get_integrated_gradients(model, example, class_to_show, baseline=None, n_steps = 5, plot_interpolation = None):
    if baseline is None:
        baseline = np.zeros((example.shape[0],3)).astype(np.float32)

    num_steps = 5
    interpolated_signal = [
            baseline + (step / num_steps) * (example - baseline)
            for step in range(num_steps + 1)
        ]
    interpolated_signal = np.array(interpolated_signal).astype(np.float32)

    if plot_interpolation is not None:
        for i in plot_interpolation:
            plt.plot(interpolated_signal[i])
            plt.plot(example)
            plt.show()
    
    grads = []
    for i, img in enumerate(interpolated_signal):
        img = tf.expand_dims(img, axis=0)
        grad = get_gradient(model, img, class_to_show)
        grads.append(grad[0])
    grads = np.array(grads)

    grads = (grads[:-1] + grads[1:]) / 2.0 # I calculate this area pointwise, for every point, and i doing it over dnum_steps
    avg_grads = grads.mean(axis=0)

    integrated_grads = (example - baseline) * avg_grads

    return integrated_grads

def get_gradient(model, input_, top_class):
    x_tensor = tf.cast(input_, tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        output = model(x_tensor)[:,top_class]

    gradients = t.gradient(output, x_tensor) # By default it is averaged so I don't get derivative w.r.t each output separately :(
    return gradients.numpy()

def plot_ith_grad(example, gradients,i, multiplier = 10):
    plt.plot(example[:,i], label ="signal")
    if gradients.ndim == 3:
        plt.plot(gradients[0,:,i] * multiplier, label = "gradient")
    else:
        plt.plot(gradients[:,i] * multiplier, label = "gradient")
    plt.legend()      