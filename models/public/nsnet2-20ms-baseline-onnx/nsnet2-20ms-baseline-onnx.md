# nsnet2-20ms-baseline-onnx

## Use Case and High-Level Description

The 'nsnet2-20ms-baseline' is a model that removes noise from an audio file by
applying a filter. The model uses the SE method based on [Recurrent Neural Network (RNN)]
<https://arxiv.org/abs/2008.06412>. Original ONNX models are provided in the
[repository](https://github.com/onnx/models).


## Specification

| Metric            | Value              |
|-------------------|--------------------|
| Type              | Speech Denoising   |
| GFLOPs            |                    |
| MParams           |                    |
| Source framework  | PyTorch\*          |

## Accuracy

Accuracy metrics are obtained on ... dataset.
| Metric | Value |
| ------ | ----- |
|        |       |
|        |       |
|        |       |

## Input

### Original Model

Logarithmic power spectrum, shape - `1,1000,161`, format is `B,T,F` where:

- `B` - batch size
- `T` - time
- `F` - frequency

### Converted Model

Logarithmic power spectrum, shape - `1,1000,161`, format is `B,T,F` where:

- `B` - batch size
- `T` - time
- `F` - frequency

## Output

### Original Model

Filter, shape - `1,1000,161`, format is `B,T,F` where:

- `B` - batch size
- `T` - time
- `F` - frequency multiplier

### Converted Model

Filter, shape - `1,1000,161`, format is `B,T,F` where:

- `B` - batch size
- `T` - time 
- `F` - frequency multiplier

## Legal Information

The original model is distributed under the following
[license](https://github.com/microsoft/DNS-Challenge/blob/master/LICENSE-CODE):

```
MIT License
Copyright (c) Microsoft Corporation.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
