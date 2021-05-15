"""
Copyright (c) 2018-2021 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ..representation import (
    SpeechDenoisingAnnotation,
    SpeechDenoisingPrediction,
)
from .metric import PerImageEvaluationMetric
from ..utils import UnsupportedPackage
import numpy as np
from math import sqrt

try:
    import python_speech_features
except ImportError as import_error:
    python_speech_features = UnsupportedPackage("python_speech_features", import_error.msg)
    
try:
    import scipy.io.wavfile
except ImportError as import_error:
    scipy.io.wavfile = UnsupportedPackage("scipy.io.wavfile", import_error.msg)
    
class CepstralDistance(PerImageEvaluationMetric):
    __provider__ = 'cepstral_distance'
    annotation_types = (SpeechDenoisingAnnotation,)
    prediction_types = (SpeechDenoisingPrediction,)
    
    def cd0(self, c1, c2):
        summary = 0
        n = min(c1.size/c1[0].size, c2.size/c2[0].size)
        for i in range(int(n)):
            p = min(c1[i].size, c2[i].size) - 1
            summary += 4.3429 * sqrt((c1[i][1]-c2[i][1])**2 +
                                 2 * sum((c1[i][2:p+1]-c2[i][2:p+1])**2))
        return summary/n

    def configure(self):
        if isinstance(python_speech_features, UnsupportedPackage):
            python_speech_features.raise_error(self.__provider__)
        if isinstance(scipy.io.wavfile, UnsupportedPackage):
            scipy.io.wavfile.raise_error(self.__provider__)
        self.values = []

    def update(self, annotation, prediction):
        coef_1 = python_speech_features.mfcc(annotation.clean_audio[0,:], 16000)
        coef_2 = python_speech_features.mfcc(prediction.denoised_audio, 16000)
        
        cepstral_distance = self.cd0(coef_1, coef_2)
        self.values.append(cepstral_distance)
        return cepstral_distance

    def evaluate(self, annotations, predictions):
        # return np.mean(np.array(self.values))
        # из за процентов
        return np.mean(np.array(self.values))/100

    def reset(self):
        self.values = []
        
class FwSegSNR(PerImageEvaluationMetric):
    __provider__ = 'fwsegSNR'
    annotation_types = (SpeechDenoisingAnnotation,)
    prediction_types = (SpeechDenoisingPrediction,)
    
    def fwsegSNR(self, clean, noisy, fs):
        # shape (161, 1000)
        weight_ratio = 0.2
        num_val = 25
        cent_freq = np.zeros((num_val,))
        bandwidth = np.zeros((num_val,))

        cent_freq[0] = 50.0000
        bandwidth[0] = 70.0000
        cent_freq[1] = 120.000
        bandwidth[1] = 70.0000
        cent_freq[2] = 190.000
        bandwidth[2] = 70.0000
        cent_freq[3] = 260.000
        bandwidth[3] = 70.0000
        cent_freq[4] = 330.000
        bandwidth[4] = 70.0000
        cent_freq[5] = 400.000
        bandwidth[5] = 70.0000
        cent_freq[6] = 470.000
        bandwidth[6] = 70.0000
        cent_freq[7] = 540.000
        bandwidth[7] = 77.3724
        cent_freq[8] = 617.372
        bandwidth[8] = 86.0056
        cent_freq[9] = 703.378
        bandwidth[9] = 95.3398
        cent_freq[10] = 798.717
        bandwidth[10] = 105.411
        cent_freq[11] = 904.128
        bandwidth[11] = 116.256
        cent_freq[12] = 1020.38
        bandwidth[12] = 127.914
        cent_freq[13] = 1148.30
        bandwidth[13] = 140.423
        cent_freq[14] = 1288.72
        bandwidth[14] = 153.823
        cent_freq[15] = 1442.54
        bandwidth[15] = 168.154
        cent_freq[16] = 1610.70
        bandwidth[16] = 183.457
        cent_freq[17] = 1794.16
        bandwidth[17] = 199.776
        cent_freq[18] = 1993.93
        bandwidth[18] = 217.153
        cent_freq[19] = 2211.08
        bandwidth[19] = 235.631
        cent_freq[20] = 2446.71
        bandwidth[20] = 255.255
        cent_freq[21] = 2701.97
        bandwidth[21] = 276.072
        cent_freq[22] = 2978.04
        bandwidth[22] = 298.126
        cent_freq[23] = 3276.17
        bandwidth[23] = 321.465
        cent_freq[24] = 3597.63
        bandwidth[24] = 346.136

        n_fft = clean.shape[0]
        filter = np.zeros((num_val, int(n_fft)))
        j = np.arange(0, n_fft)

        for i in range(num_val):
            cf = (cent_freq[i] / (fs / 2)) * (n_fft)
            bw = (bandwidth[i] / (fs / 2)) * (n_fft)
            norm_factor = np.log(bandwidth[0]) - np.log(bandwidth[i])
            filter[i, :] = np.exp(-11 * (((j - np.floor(cf)) / bw) ** 2) + norm_factor)

        clean = np.abs(clean)
        clean = clean / np.sum(clean, 0)
        noisy = np.abs(noisy)
        noisy = noisy / np.sum(noisy, 0)
        clean = filter.dot(clean)
        noisy = filter.dot(noisy)
        error = (clean - noisy) ** 2
        eps = np.finfo(np.float).eps
        error[error < eps] = eps
        W = clean ** weight_ratio
        SNR = 10 * np.log10((clean ** 2) / error)
        fwSNR = np.sum(W * SNR, 0) / np.sum(W, 0)
        return np.mean(fwSNR)

    
    def configure(self):
        self.values = []

    def update(self, annotation, prediction):
        fwsegsnr = self.fwsegSNR(prediction.get_spectrum(annotation.clean_audio), prediction.get_spectrum(np.expand_dims(prediction.denoised_audio, axis=0)), 16000)
        self.values.append(fwsegsnr)
        return fwsegsnr

    def evaluate(self, annotations, predictions):
        # return np.mean(np.array(self.values))
        # из за процентов
        return np.mean(np.array(self.values))/100

    def reset(self):
        self.values = []
        
class STOI(PerImageEvaluationMetric):
    __provider__ = 'stoi'
    annotation_types = (SpeechDenoisingAnnotation,)
    prediction_types = (SpeechDenoisingPrediction,)
    
    def stoi(self, x, y, fs = 10000):
        if x.shape != y.shape:
            raise Exception('signals should have the same length')
        
        N = 30
        OBM = self.create_OBM(fs) 

        x, y = self.remove_silent(x, y)

        x_spec = self.stft(x).transpose()
        y_spec = self.stft(y).transpose()

        x_ = np.sqrt(np.matmul(OBM, np.square(np.abs(x_spec))))
        y_ = np.sqrt(np.matmul(OBM, np.square(np.abs(y_spec))))

        x_segments = np.array([x_[:, m - N:m] 
                        for m in range(N, x_.shape[1] + 1)])
        y_segments = np.array([y_[:, m - N:m] 
                        for m in range(N, x_.shape[1] + 1)])

        x_norm = self.normalize(x_segments)
        y_norm = self.normalize(y_segments)

        return np.sum(x_norm * y_norm / N) / y_norm.shape[0]

    
    def create_OBM(self, fs):   
        arr = np.array(range(15))
        cf = np.power(2 ** (1 / 3), arr) * 150
        low = 150 * np.power(2, (2 * arr - 1) / 6)
        high = 150 * np.power(2, (2 * arr + 1) / 6)
        f = np.linspace(0, fs, 513)
        f = f[:257]
        obm = np.zeros((15, len(f)))

        for i in range(len(cf)):
            l_ii = np.argmin(np.square(f - low[i]))
            h_ii = np.argmin(np.square(f - high[i]))
            obm[i, l_ii:h_ii] = 1
        return obm


    def stft(self, x):
        w = np.hanning(258)[1: -1]
        out = np.array([np.fft.rfft(w * x[i:i + 256], n=512)
                        for i in range(0, len(x) - 256, 128)])
        return out

    def normalize(self, x):
        x_normed = x + self.EPS * np.random.standard_normal(x.shape)
        x_normed -= np.mean(x_normed, axis=-1, keepdims=True)
        x_inv = 1. / np.sqrt(np.sum(
                    np.square(x_normed), axis=-1, keepdims=True))
        x_diags = np.array(
            [np.diag(x_inv[i].reshape(-1)) for i in range(x_inv.shape[0])])
        x_normed = np.matmul(x_diags, x_normed)
        x_normed += self.EPS * np.random.standard_normal(x_normed.shape)
        x_normed -= np.mean(x_normed, axis=1, keepdims=True)
        x_inv = 1. / np.sqrt(np.sum(
                    np.square(x_normed), axis=1, keepdims=True))
        x_diags = np.array(
            [np.diag(x_inv[i].reshape(-1)) for i in range(x_inv.shape[0])])
        x_normed = np.matmul(x_normed, x_diags)
        return x_normed

    def remove_silent(self, x, y):
        w = np.hanning(258)[1:-1]
        x_frames = np.array([w * x[i:i + 256] 
                    for i in range(0, len(x) - 256, 128)])
        y_frames = np.array([w * y[i:i + 256] 
                    for i in range(0, len(x) - 256, 128)])

        x_energies = 20 * np.log10(np.linalg.norm(x_frames, axis=1) + self.EPS)
        mask = (np.max(x_energies) - 40 - x_energies) < 0

        x_frames = x_frames[mask]
        y_frames = y_frames[mask]

        x_out = np.zeros((len(x_frames) - 1) * 128 + 256)
        y_out = np.zeros((len(x_frames) - 1) * 128 + 256)

        for i in range(x_frames.shape[0]):
            x_out[range(i * 128, i * 128 + 256)] += x_frames[i, :]
            y_out[range(i * 128, i * 128 + 256)] += y_frames[i, :]

        return x_out, y_out
    
    def configure(self):
        self.EPS = np.finfo("float").eps
        self.values = []

    def update(self, annotation, prediction):
        stoi_metric = self.stoi(annotation.clean_audio[0,:], prediction.denoised_audio, 16000)

        self.values.append(stoi_metric)
        return stoi_metric

    def evaluate(self, annotations, predictions):
        return np.mean(np.array(self.values))

    def reset(self):
        self.values = []

