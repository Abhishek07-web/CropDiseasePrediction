import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform as BaseGlorotUniform

class GlorotUniformCompat(BaseGlorotUniform):
    def __init__(self, seed=None, input_axes=None, output_axes=None):
        super().__init__(seed=seed)

    def get_config(self):
        config = super().get_config()
        config.update({'input_axes': None, 'output_axes': None})
        return config

print('TF version', tf.__version__)
try:
    model = tf.keras.models.load_model('models/crop_disease_model.h5', compile=False, custom_objects={'GlorotUniform': GlorotUniformCompat})
    print('Loaded model successfully')
except Exception:
    import traceback
    traceback.print_exc()
