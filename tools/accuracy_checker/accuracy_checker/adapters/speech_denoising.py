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

import numpy as np
from ..adapters import Adapter
from ..representation import SpeechDenoisingPrediction


class SpeechDenoisingAdapter(Adapter):
    __provider__ = 'speech_denoising'
    prediction_types = (SpeechDenoisingPrediction, )

    def process(self, raw, identifiers, frame_meta):

        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)

        output = raw_outputs['output']

        for i, identifier in enumerate(identifiers):
            result.append(SpeechDenoisingPrediction(identifier, raw[i:i+1]))
        return result
