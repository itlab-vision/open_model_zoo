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
        return np.mean(np.array(self.values))

    def reset(self):
        self.values = []
        
class FwSegSNR(PerImageEvaluationMetric):
    __provider__ = 'fwsegSNR'
    annotation_types = (SpeechDenoisingAnnotation,)
    prediction_types = (SpeechDenoisingPrediction,)
    
    def fwsegSNR(self, clean, noisy, fs):
        num_seg = clean.shape[1]
        num_freq = clean.shape[0]
        S = 0
        B = self.B_weight(num_seg, fs)
        for i in range(num_freq):
            Num = 0
            Denom = 0
            for j in range(num_seg):
                Num += B[j] * np.log10((clean[i][j] ** 2) / ((clean[i][j] - noisy[i][j]) ** 2))
                Denom += B[j]
            S += Num / Denom
        return (10 / num_seg) * S

    def B_weight(self, num_seg, fs):
        B = np.zeros(num_seg)
        for i in range(num_seg):
            f = (i + 1) * (fs / 2) / num_seg
            Rb = ((12194 ** 2) * (f ** 3)) / (((f ** 2) + (20.6 ** 2)) * ((f ** 2) + (12194 ** 2)) * (((f ** 2) + (158.5 ** 2)) ** 0.5))
            B[i] = 0.17 + 20 * np.log10(Rb)
        return B
    
    def configure(self):
        self.values = []

    def update(self, annotation, prediction):
        fwsegsnr = self.fwsegSNR(annotation.get_spectrum(annotation.clean_audio), prediction.denoised_spectrum, 16000)
        
        self.values.append(fwsegsnr)
        return fwsegsnr

    def evaluate(self, annotations, predictions):
        return np.mean(np.array(self.values))

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

