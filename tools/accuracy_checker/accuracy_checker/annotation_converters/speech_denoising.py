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
from ..utils import read_csv, get_path, check_file_existence


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
        self.data_dir = self.get_value_from_config('data_dir')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100,**kwargs):
        annotation = []
        content_errors = [] if check_content else None
        
        data_dir_files = self.data_dir.iterdir()
        for file_in_dir in data_dir_files:
            annotation.append(SpeechDenoisingAnnotation(file_in_dir))   
            if progress_callback is not None and audio_id % progress_interval == 0:
                progress_callback(audio_id / num_iterations * 100)

        if check_content:
            if annotation is [] :
                content_errors.append('Folder is empty')

        return ConverterReturn(annotation, None, content_errors)
