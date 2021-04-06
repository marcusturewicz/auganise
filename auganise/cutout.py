# Improved Regularization of Convolutional Neural Networks with Cutout
# arXiv:1708.04552

import tensorflow_addons as tfa
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras import backend

class RandomCutout(PreprocessingLayer):
    """Randomly cuts out a section of the images to target mask height and width.
    This layer will cut out the same section for all the images in the same batch.
    By default, random cut out is only applied during training. If you need to
    apply random cut out at inference time, set `training` to True when calling the layer.
    Input shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.
    Output shape:
    4D tensor with shape:
    `(samples, target_height, target_width, channels)`.
    Args:
    mask_size: Integer, the width and height of the cut out region.
    constant_value: Float, the value to fill the cut out region with.
    seed: Integer. Used to create a random seed.
    Raise:
        ValueError: if mask is not of type int or a multiple of 2.
    """

    def __init__(self, mask_size, seed=None, name=None, **kwargs):
        if not isinstance(mask_size, int) and not mask_size // 2 == 0:
            raise ValueError(f'RandomCutout layer {name} received an invalid mask '
                'argument {mask}. Must be of type int and a mulitple of 2.')

        self.mask_size = mask_size
        self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        super(RandomCutout, self).__init__(name=name, **kwargs)
        # TODO: is the following line required?
        # base_preprocessing_layer._kpl_gauge.get_cell('RandomCutout').set(True)

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()

        def random_cutout_inputs():
            """Cutout inputs using tfa.image.random_cutout"""
            return tfa.image.random_cutout(inputs, mask_size=(self.mask_size, self.mask_size), constant_values=0, seed=self.seed)

        output = control_flow_util.smart_cond(training, random_cutout_inputs, lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'mask_size': self.mask_size,
            'seed': self.seed,
        }
        base_config = super(RandomCutout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
