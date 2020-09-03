import tensorflow as tf

def generate_layers_linear(n_in, n_out, n_layers, k):
    layers = [n_in]
    units = n_layers * k + n_out
    for i in range(n_layers):
        units -= k
        layers.append(units)
    return layers


class Dqn(tf.keras.Model):
    def __init__(self, layers):
        super(Dqn, self).__init__()

        self.model = tf.keras.Sequential()

        for i, units in enumerate(layers):
            self.model.add(tf.keras.layers.Dense(units=units, input_shape=(units,)))
            if i < len(layers):
                self.model.add(tf.keras.layers.BatchNormalization())
                self.model.add(tf.keras.layers.LeakyReLU())

    def call(self, inputs):
        _len = len(self.model.layers)
        for i, layer in enumerate(self.model.layers):
            inputs = layer(inputs)
            if i == _len - 1 and self._log:
                self.logs.append(inputs)
        return inputs
