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

from ..config import BoolField, BaseField, NumberField, ConfigError, StringField
from ..postprocessor import Postprocessor
from ..representation import SpeechDenoisingPrediction, SpeechDenoisingAnnotation
from ..utils import UnsupportedPackage
try:
    import scipy.signal as dsp
except ImportError as import_error:
    mask_util = UnsupportedPackage('scipy', import_error.msg)


windows = {
    'hann': np.hanning,
    'hamming': np.hamming,
    'blackman': np.blackman,
    'bartlett': np.bartlett,
    'none': None,
}

class MelSpectrogramToAudio(Postprocessor):
    __provider__ = 'mel_spectrogram_to_audio'

    prediction_types = (SpeechDenoisingPrediction, )
    annotation_types = (SpeechDenoisingAnnotation, )


    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'window_size': NumberField(optional=True, value_type=float, default=0.02,
                                       description="Size of frame in time-domain, seconds"),
            'window_stride': NumberField(optional=True, value_type=float, default=0.01,
                                         description="Intersection of frames in time-domain, seconds"),
            'window': StringField(
                choices=windows.keys(), optional=True, default='hann', description="Weighting window type"
            ),
            'n_fft': NumberField(optional=True, value_type=int, description="FFT base"),
            'n_filt': NumberField(optional=True, value_type=int, default=80, description="Number of MEL filters"),
            'splicing': NumberField(optional=True, value_type=int, default=1,
                                    description="Number of sequentially concastenated MEL spectrums"),
            'sample_rate': NumberField(optional=True, value_type=float, description="Audio samplimg frequency, Hz"),
            'pad_to': NumberField(optional=True, value_type=int, default=0, description="Desired length of features"),
            'preemph': NumberField(optional=True, value_type=float, default=0.97, description="Preemph factor"),
            'log': BoolField(optional=True, default=True, description="Enables log() of MEL features values"),
            'use_deterministic_dithering': BoolField(optional=True, default=True,
                                                     description="Applies determined dithering to signal spectrum"),
            'dither': NumberField(optional=True, value_type=float, default=0.00001, description="Dithering value"),
            'no_delay': BoolField(optional=True, default=False, description="Remove first delay frame from stft result")
        })
        return params

    def configure(self):
        self.window_size = self.get_value_from_config('window_size')
        self.window_stride = self.get_value_from_config('window_stride')
        self.n_fft = self.get_value_from_config('n_fft')
        self.window_fn = windows.get(self.get_value_from_config('window'))
        self.preemph = self.get_value_from_config('preemph')
        self.nfilt = self.get_value_from_config('n_filt')
        self.sample_rate = self.get_value_from_config('sample_rate')
        self.log = self.get_value_from_config('log')
        self.pad_to = self.get_value_from_config('pad_to')
        self.frame_splicing = self.get_value_from_config('splicing')
        self.use_deterministic_dithering = self.get_value_from_config('use_deterministic_dithering')
        self.dither = self.get_value_from_config('dither')
        self.no_delay = self.get_value_from_config('no_delay')

        self.normalize = 'per_feature'
        self.lowfreq = 0
        self.highfreq = None
        self.max_duration = 16.7
        self.pad_value = 0
        self.mag_power = 2.0
        self.use_deterministic_dithering = True
        self.dither = 1e-05
        self.log_zero_guard_type = 'add'
        self.log_zero_guard_value = 2 ** -24


    def process_image(self, annotation, prediction):
        
        print('I am performed!')
        return annotation, prediction

