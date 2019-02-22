import tensorflow as tf
from tflearn.layers.conv import global_avg_pool

class SE_layer():
    def __init__(self, x, training=True):
        self.training = training

    def Global_Average_Pooling(self, x):
        return global_avg_pool(x, name='Global_avg_pooling')

    def Fully_connected(self, x, units=3, layer_name='fully_connected') :
        with tf.name_scope(layer_name) :
            return tf.layers.dense(inputs=x, use_bias=False, units=units)

    def Relu(self, x):
        return tf.nn.relu(x)
    def Sigmoid(self, x) :
        return tf.nn.sigmoid(x)

    def squeeze_excitation_layer(self, input_x, ratio, layer_name):
        with tf.name_scope(layer_name) :
            squeeze = self.Global_Average_Pooling(input_x)

            excitation = self.Fully_connected(squeeze, units=int(input_x.shape[3])/ratio, layer_name=layer_name+'_fully_connected1')
            excitation = self.Relu(excitation)
            excitation = self.Fully_connected(excitation, units=int(input_x.shape[3]), layer_name=layer_name+'_fully_connected2')
            excitation = self.Sigmoid(excitation)
            dim3 = int(input_x.shape[3])
            excitation = tf.reshape(excitation, [-1,1,1, dim3])
            scale = input_x * excitation

            return scale


if __name__=="__main__":
    input_data = tf.random_uniform([2, 70, 80, 3], 0, 255)
    semodule = SE_layer(input_data)
    output = semodule.squeeze_excitation_layer(input_data, 1, 'first')
    print(output.shape)
