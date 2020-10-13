import logging as log
from abc import ABC, abstractmethod
import sys
import os
import numpy as np

from .features import calcFeat, calcSpec, spec2sig
from openvino.inference_engine import  IECore

class SpeechDenoiser(ABC):
    @staticmethod
    def create(args):
        if 'nsnet2-20ms-baseline' in args.model:
            return DeepNoiseSuppression(args.model, args.device)
        else:
            raise Exception('Error: wrong name')

    @abstractmethod
    def denoise(self, data):
         '''Perform Noise Suppression'''

class DeepNoiseSuppression(SpeechDenoiser):
    chunk = 1
    def __init__(self, model, device):
        log.basicConfig(format='[ %(levelname)s ] %(message)s', level= log.INFO, stream=sys.stdout)
        self.cfg = {
            'winlen'   : 0.02,
            'hopfrac'  : 0.5,
            'fs'       : 16000,
            'mingain'  : -80,
            'feattype' : 'LogPow'
        }
        self.mingain = 10**(self.cfg['mingain']/20)
        # Plugin initialization
        ie = IECore()
        # log.info("Loading network")
        net = ie.read_network(model, os.path.splitext(model)[0] + ".bin")
        self.input_blob = next(iter(net.input_info)) # ?
        self.output_blob = 'Sigmoid_31'
        assert len(net.input_info) == 1, "One input is expected"
        # Loading model to the plugin
        # log.info("Loading model to the plugin")
        self.exec_net = ie.load_network(network=net, device_name= device)

    def preprocessing(self, data):
        self.inputSpec = calcSpec(data, self.cfg)
        inputFeature = calcFeat(self.inputSpec, self.cfg)
        # shape: [batch x time x freq]
        inputFeature = np.expand_dims(np.transpose(inputFeature), axis=0)
        return inputFeature

    def postprocessing(self, out):
        Gain = np.transpose(out)
        Gain = np.clip(Gain, a_min=self.mingain, a_max=1.0)
        outSpec = np.expand_dims(self.inputSpec, axis=2) * Gain

        # go back to time domain
        out = spec2sig(outSpec, self.cfg)
        return out

    def denoise(self, data):
        # log.info("Preprocessing input")
        inputFeature = self.preprocessing(data)
        # Calculate network output
        # log.info("Starting inference")
        out = self.exec_net.infer({self.input_blob: inputFeature})[self.output_blob]
        # print(out.shape)
        # log.info("Inference successfull")
        # log.info("Output shape is " + str(out.shape))
        # log.info("Postprocessing output")
        result = self.postprocessing(out)
        return result

