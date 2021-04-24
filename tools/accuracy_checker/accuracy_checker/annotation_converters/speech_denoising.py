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

from .format_converter import ConverterReturn, DirectoryBasedAnnotationConverter
from ..config import PathField
from ..representation import SpeechDenoisingAnnotation
from ..preprocessor import SamplesToFloat32, ResampleAudio, ClipAudio, AudioToMelSpectrogram
from ..data_readers import DataRepresentation
from ..utils import read_csv, get_path, check_file_existence
from ..utils import UnsupportedPackage
try:
    from pathlib import Path
except ImportError as import_error:
    Path = UnsupportedPackage("pathlib", import_error.msg)
try:
    import wave
except ImportError as import_error:
    wave = UnsupportedPackage("wave", import_error.msg)
try:
    import numpy as np
except ImportError as import_error:
    np = UnsupportedPackage("numpy", import_error.msg)

class SpeechDenoisingFormatConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'speech_denoising'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
             'data_dir': PathField(
                 is_directory=True, optional=False,
             )
        })

        return parameters

    def configure(self):
        if isinstance(wave, UnsupportedPackage):
            wave.raise_error(self.__provider__)
        if isinstance(np, UnsupportedPackage):
            np.raise_error(self.__provider__)
        if isinstance(Path, UnsupportedPackage):
            Path.raise_error(self.__provider__)
        self.data_dir = self.get_value_from_config('data_dir')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100,**kwargs):
        annotation = []
        content_errors = [] if check_content else None
        
        cleans = []
        clean_spectrums = []
        noisy_spectrums= []
        files_in_dir = []
        
        metadata = {'sample_rate' : 16000}
        ra_config = {'sample_rate' : 16000}
        ca_config = {'duration': '160000 samples', 'overlap': '60000 samples'}
        atms_config = {
            'window_size': 0.02,
            'window_stride': 0.01,
            'window': 'hamming',
            'n_fft': 320,
            'n_filt': 161,
            'splicing': 1,
            'sample_rate': 16000,
            'no_delay': True
        }
        data_dir_files = self.data_dir.iterdir()
        for file_in_dir in data_dir_files:
            with wave.open(str(file_in_dir), "rb") as wav:
                sample_width = wav.getsampwidth()
                nframes = wav.getnframes()
                clean_audio = wav.readframes(nframes)
                if {1: np.uint8, 2: np.int16}.get(sample_width):
                    clean_audio = np.frombuffer(clean_audio, dtype={1: np.uint8, 2: np.int16}[sample_width])
                else:
                    raise RuntimeError("Reader {} couldn't process file {}: unsupported sample width {}"
                                   "(reader only supports {})"
                                   .format('wave', file_in_dir,
                                           sample_width, [*{1: np.uint8, 2: np.int16}.keys()]))
                clean_audio = clean_audio.reshape(-1, wav.getnchannels()).T
                
                image = DataRepresentation(clean_audio, metadata)
                image = SamplesToFloat32({}).process(image)
                image = ResampleAudio(ra_config).process(image)
                image = ClipAudio(ca_config).process(image)
                cleans.append(image.data[0])
                image = AudioToMelSpectrogram(atms_config).process(image)
                image.data = np.transpose(image.data[0,:,:])
                clean_spectrums.append(image.data)
                files_in_dir.append(file_in_dir)
                
            if progress_callback is not None and audio_id % progress_interval == 0:
                progress_callback(audio_id / num_iterations * 50)
        
        data_dir_files = Path(str(self.data_dir)[:-11] + 'noisy\\audio').iterdir()
        for file_in_dir in data_dir_files:
            with wave.open(str(file_in_dir), "rb") as wav:
                sample_width = wav.getsampwidth()
                nframes = wav.getnframes()
                noisy_audio = wav.readframes(nframes)
                if {1: np.uint8, 2: np.int16}.get(sample_width):
                    noisy_audio = np.frombuffer(noisy_audio, dtype={1: np.uint8, 2: np.int16}[sample_width])
                else:
                    raise RuntimeError("Reader {} couldn't process file {}: unsupported sample width {}"
                                   "(reader only supports {})"
                                   .format('wave', file_in_dir,
                                           sample_width, [*{1: np.uint8, 2: np.int16}.keys()]))
                noisy_audio = noisy_audio.reshape(-1, wav.getnchannels()).T
            
                image = DataRepresentation(noisy_audio, metadata)
                image = SamplesToFloat32({}).process(image)
                image = ResampleAudio(ra_config).process(image)
                image = ClipAudio(ca_config).process(image)
                image = AudioToMelSpectrogram(atms_config).process(image)
                image.data = np.transpose(image.data[0,:,:])
                noisy_spectrums.append(image.data)
        
            if progress_callback is not None and audio_id % progress_interval == 0:
                progress_callback(audio_id / num_iterations * 50)
                
        for i in range(len(files_in_dir)):
            annotation.append(SpeechDenoisingAnnotation(files_in_dir[i], cleans[i], clean_spectrums[i], noisy_spectrums[i]))
            
        if check_content:
            if annotation is [] :
                content_errors.append('Folder is empty')

        return ConverterReturn(annotation, None, content_errors)
