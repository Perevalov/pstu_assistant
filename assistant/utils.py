import keras.backend as K
from keras.layers.pooling import _GlobalPooling1D


def dummy_loss(y_true, y_pred):
    return y_pred

class MaskedGlobalMaxPooling1D(_GlobalPooling1D):
    
    def __init__(self, **kwargs):
        self.support_mask = True
        super(MaskedGlobalMaxPooling1D, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(MaskedGlobalMaxPooling1D, self).build(input_shape)
        self.feat_dim = input_shape[2]

    def call(self, x, mask=None):
        ans = K.max(x, axis=1)
        return ans

    def compute_mask(self, input_shape, input_mask=None):
        return None