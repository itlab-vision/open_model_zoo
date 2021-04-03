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
        #желательно
        #coef_1 = python_speech_features.mfcc(annotation.clean_audio, annotation.rate)
        #coef_2 = python_speech_features.mfcc(prediction.denoised_audio, prediction.rate)
    
        rate, sig = scipy.io.wavfile.read(annotation.identifier)
        coef_1 = python_speech_features.mfcc(sig, rate)
        # ввиду отсутствия постпроцессинга
        coef_2 = python_speech_features.mfcc(sig * 2, rate)
        
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
    
    def fwsegSNR(self, clean, data): # clean и data спектрограммы
        num_seg = clean.shape[1] # Количество окон в спектрограмме
        num_freq = clean.shape[0] # Количество частот в спектрограмме
        S = 0
        B = np.ones(num_freq) # Веса полос частот
        # Здесь должны быть определённые коэфиценты в зависимости от восприятия слуховой системой человека
        for i in range(num_seg):
            Num = 0
            Denom = 0
            for j in range(num_freq):
                Num += B[j] * np.log10((clean[i][j] ** 2) / ((clean[i][j] - data[i][j]) ** 2))
                Denom += B[j]
            S += Num / Denom
        return (10 / num_seg) * S

    def configure(self):
        self.values = []

    def update(self, annotation, prediction):
        #желательно
        #fwsegsnr = self.fwsegSNR(annotation.spectr, prediction.spectr)
    
        # ввиду отсутствия annotation.spectr и prediction.spectr
        from random import random
        fwsegsnr = random()
        
        self.values.append(fwsegsnr)
        return fwsegsnr

    def evaluate(self, annotations, predictions):
        return np.mean(np.array(self.values))

    def reset(self):
        self.values = []

