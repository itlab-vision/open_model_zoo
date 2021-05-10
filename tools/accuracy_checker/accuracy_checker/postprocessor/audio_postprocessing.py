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
import inspect
import wave
from scipy.io import wavfile
import soundfile as sf
from openvino.inference_engine import  IECore

from accuracy_checker.config import BoolField, BaseField, NumberField, ConfigError, StringField
from accuracy_checker.postprocessor import Postprocessor
from accuracy_checker.representation import SpeechDenoisingPrediction, SpeechDenoisingAnnotation
from accuracy_checker.utils import UnsupportedPackage
from accuracy_checker.preprocessor import AudioToSpectrogram, ClipAudio
from accuracy_checker.data_readers import WavReader, DataRepresentation

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
            'sample_rate': NumberField(optional=True, value_type=float,default=16000, description="Audio samplimg frequency, Hz"),
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
        self.n_filt = self.get_value_from_config('n_filt')
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
        
        self.ca_config = {'duration': '160000 samples', 'overlap': '60000 samples'}
        self.metadata = {'sample_rate' : 16000}
    
    def istft(self,X, N_fft, win, N_hop):
        """
        inverse short-time Fourier transform
            X 			Spectra [frequency x frames x channels]
            N_fft 		FFT size (samples)
            win 		window,  len(win) <= N_fft
            N_hop 		hop size (samples)
        """
        specsize = X.shape[0]
        N_frames = X.shape[1]
        if X.ndim < 3:
            X = X[:,:,np.newaxis]
        M = X.shape[2]
        N_win = len(win)
        Nx = N_hop*(N_frames-1) + N_win
        win_M = np.outer(win,np.ones((1,M)))
        x = np.zeros((Nx,M))
        for nn in range(0,N_frames):
            X_frame = np.squeeze(X[:,nn,:])
            x_win = np.fft.irfft(X_frame, N_fft, axis=0)
            x_win = x_win.reshape(N_fft,M)
            x_win = win_M * x_win[0:N_win,:]
            idx1 = int(nn*N_hop); idx2 = int(idx1+N_win)
            x[idx1:idx2,:] = x_win + x[idx1:idx2,:]
        if M==1:
            x = np.squeeze(x)
        return x

    def convertSpecToAudio(self,out,input):
        Gain = np.transpose(out)
        Gain = np.clip(Gain, a_min=10**(-80.0/20), a_max=1.0)
        outSpec = np.expand_dims(input, axis=2) * Gain
        N_win = int(float(self.window_size)*float(self.sample_rate))
        win = np.sqrt(np.hanning(N_win))
        x = self.istft(outSpec, self.n_fft, win, self.n_filt)
        return x

    def process_image(self, annotation,prediction):
        spec= SpeechDenoisingAnnotation.get_spectrum(annotation[0].noisy_audio)
        filter= prediction[0]._filter[0]['output']
        data = np.expand_dims(self.convertSpecToAudio(filter,spec),axis=0)
        image = DataRepresentation(data, self.metadata)
        image = ClipAudio(self.ca_config).process(image)
        prediction[0].denoised_audio = image.data[0][0,:]
        prediction[0].denoised_spectrum = SpeechDenoisingAnnotation.get_spectrum(np.expand_dims(prediction[0].denoised_audio, axis=0))
        return annotation, prediction
