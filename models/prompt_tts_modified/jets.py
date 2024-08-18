# Copyright 2023, YOUDAO
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from models.prompt_tts_modified.model_open_source import PromptTTS
from models.hifigan.models import Generator as HiFiGANGenerator

from models.hifigan.get_random_segments import get_random_segments, get_segments


class JETSGenerator(nn.Module):
    def __init__(self, config) -> None:

        super().__init__()
        
        self.upsample_factor= int(np.prod(config.model.upsample_rates))

        self.segment_size = config.segment_size

        self.am = PromptTTS(config)

        self.generator = HiFiGANGenerator(config.model)

        self.config=config


    def forward(self, inputs_ling, input_lengths, inputs_speaker, inputs_style_embedding, inputs_content_embedding, alpha=1.0, cut_flag=True):

        outputs = self.am(inputs_ling, input_lengths, inputs_speaker, inputs_style_embedding, inputs_content_embedding, alpha)
        
        z_segments = outputs["dec_outputs"].transpose(1,2)
        z_start_idxs=None
        segment_size=self.segment_size

        wav = self.generator(z_segments)

        outputs["wav_predictions"] = wav
        outputs["z_start_idxs"]= z_start_idxs
        outputs["segment_size"] = segment_size
        return outputs
