"""
Copyright (c) 2018-2020 Intel Corporation

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

import re

from ._reid_common import check_dirs, read_directory
from .format_converter import DirectoryBasedAnnotationConverter, ConverterReturn
from ..representation import ReIdentificationAnnotation

MARS_IMAGE_PATTERN = re.compile(r'([\d]+)C(\d)')


class MARSConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'mars'
    annotation_types = (ReIdentificationAnnotation, )

    def convert(self, check_content=False, **kwargs):
        gallery = self.data_dir / 'bbox_test'
        query = self.data_dir / 'query'

        check_dirs((gallery, query), self.data_dir)
        gallery_images, gallery_pids = read_directory(gallery, query=False, image_pattern=MARS_IMAGE_PATTERN)
        query_images, query_pids = read_directory(query, query=True, image_pattern=MARS_IMAGE_PATTERN)

        return ConverterReturn(gallery_images + query_images, {'num_identities': len(gallery_pids | query_pids)}, None)
