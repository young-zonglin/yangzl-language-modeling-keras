**Warning**: There are some problems with the implementation of this language model project. You should use a multi-input and multi-output network structure instead of multiple input and single output.

**Warning**: The implementation of the generator is not efficient enough, the GPU will be idle periodically!

**Recommendation**: Use PyTorch's [fairseq](https://github.com/pytorch/fairseq) nlp library or TensorFlow's [tensor2tensor](https://github.com/tensorflow/tensor2tensor) library.

## Keras language modeling
My first language model implemented by Keras with LSTM in Python.

## Get started
- The main entry is at `train.py`. 
- Train model using `train.py`. 
- The configuration of network is in `network_conf.py`.
