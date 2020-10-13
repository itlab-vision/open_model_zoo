import logging as log
import os
import sys
import time
import argparse
import soundfile as sf
import sounddevice as sd
import numpy as np

from openvino.inference_engine import IECore
from speech_denoising_demo.model import SpeechDenoiser

def build_argparser():
    parser = argparse.ArgumentParser(description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                        help= 'Required. Path to .XML file with pre-trained model.')
    args.add_argument('-i', '--input', type = str, required = True,
                        help = 'Required. Audiofile to process. Default value is "mic" to work with default microphone')
    args.add_argument('-o', '--output', type = str, default = './',
                        help = 'Optional. Path to output dir.')
    args.add_argument('-d', '--device', type = str, default='CPU',
                        help = 'Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is '
                             'acceptable. The demo will look for a suitable plugin for the device specified. '
                             'Default value is CPU')
    args.add_argument('-no_play',  action='store_true', help = 'Optional. Do not play inference results.')

    return parser

#def preprocess_audio(input)

def main():
    """ Main function. """
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level= log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info('Create denoiser')
    denoiser = SpeechDenoiser.create(args)
    fs = 16000 # default sample rate for input

    if args.input == 'mic':
        log.info('Start recording')
        duration = 5  # seconds
        data = sd.rec(int(duration * fs), samplerate = fs, channels=1)
        sd.wait()
        log.info('Saving recording')
        sf.write('recording.wav', data, fs)
        log.info('Recording was written to ' + str(os.path.abspath(os.getcwd())) + '\\recording.wav')
    else:
        data, fs = sf.read(args.input)
        data = np.reshape(data, (-1,1))
        duration = int(data.shape[0]/fs)
        print(fs)
    
    # divide audio into blocks and process them one by one
    nblocks = int(duration / denoiser.chunk)
    block_size = int(denoiser.chunk*fs)
    print(nblocks)
    print(block_size)
    print(data.shape)
    res = np.zeros(shape=(0,1))


    for i in range(nblocks):
        input_data = data[i*block_size:(i+1)*block_size, 0]
        out = denoiser.denoise(input_data).reshape(-1, 1)
        if i > 0:
            print(i)
            tmpres = res[-160:]
            tmpres += out[:160]
            res[-160:] = tmpres
            res = np.concatenate((res, out[160:out.shape[0]]))
        else:
            res = np.concatenate((res, out[:out.shape[0]]))

    print('Number of blocks was', nblocks)

    log.info('Playing result')
    print(res.shape)
    sd.play(res, fs)
    status = sd.wait()

    log.info('Saving result')
    sf.write('res.wav', res, fs)
    log.info('Result was written to ' + str(os.path.abspath(os.getcwd())) + '\\res.wav') # TODO: add output parameter
    # at least two cases : 1) upload audio, process it an than play. 2) record, process, play

if __name__ == '__main__':
    sys.exit(main() or 0)
