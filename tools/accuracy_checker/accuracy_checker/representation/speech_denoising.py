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

from .base_representation import BaseRepresentation
from ..data_readers import DataRepresentation
from ..preprocessor import AudioToSpectrogram
import numpy as np

class SpeechDenoisingRepresentation(BaseRepresentation):
    pass


class SpeechDenoisingAnnotation(SpeechDenoisingRepresentation):
    def __init__(self, identifier, clean_audio, noisy_audio):
        super().__init__(identifier)
        self.clean_audio = clean_audio
        self.noisy_audio = noisy_audio
    @staticmethod
    def get_spectrum(audio):
        ats_config = {
            'window_size': 0.02,
            'window_stride': 0.01,
            'window': 'hamming',
            'n_fft': 320,
            'n_filt': 161,
            'splicing': 1,
            'sample_rate': 16000,
            'no_delay': True
        }
        spec = AudioToSpectrogram(ats_config).calcSpec(audio)
        return spec
        
        


class SpeechDenoisingPrediction(SpeechDenoisingRepresentation):
    def __init__(self, identifier, _filter, denoised_audio = None):
        super().__init__(identifier)
        self._filter = _filter
        self.denoised_audio = denoised_audio
