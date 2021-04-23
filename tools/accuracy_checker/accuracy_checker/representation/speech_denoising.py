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


class SpeechDenoisingRepresentation(BaseRepresentation):
    pass


class SpeechDenoisingAnnotation(SpeechDenoisingRepresentation):
    def __init__(self, identifier, clean_audio, clean_spectrum = None, noisy_spectrum = None):
        super().__init__(identifier)
        self.clean_audio = clean_audio
        self.clean_spectrum = clean_spectrum
        self.noisy_spectrum = noisy_spectrum


class SpeechDenoisingPrediction(SpeechDenoisingRepresentation):
    def __init__(self, identifier, _filter, denoised_audio = None, denoised_spectrum = None):
        super().__init__(identifier)
        self._filter = _filter
        self.denoised_audio = denoised_audio
        self.denoised_spectrum = denoised_spectrum
