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

#try:
#    import editdistance
#except ImportError as import_error:
#    editdistance = UnsupportedPackage("editdistance", import_error.msg)

class CepstralDistance(PerImageEvaluationMetric):
    __provider__ = 'cepstral_distance'
    annotation_types = (SpeechDenoisingAnnotation,)
    prediction_types = (SpeechDenoisingPrediction,)

    def configure(self):
#        if isinstance(editdistance, UnsupportedPackage):
#            editdistance.raise_error(self.__provider__)
        self.values = []

    def update(self, annotation, prediction):
        from random import random
        cepstral_distance = random()
        self.values.append(cepstral_distance)
        return cepstral_distance

    def evaluate(self, annotations, predictions):
        return np.mean(np.array(self.values))

    def reset(self):
        self.values = []
