# coding=utf-8
import tensorflow as tf
from config import cfg
from net_tf import Net

"""
this is a demo. I can through this to quick build a net.
"""
class TrainNet(Net):
    def __init__(self):
        super(TrainNet, self).__init__()

        # input
        # [N, 4]
        self.pc_input = tf.placeholder(tf.float32, shape=[None, 4], name='pointcloud')
        # [ΣK, 200, 6]
        self.voxel_feature = tf.placeholder(tf.float32, [None, cfg.VOXEL_POINT_COUNT, 6], name='grid')
        # [ΣK, 3], each row stores (batch, d, w)
        self.coordinate = tf.placeholder(tf.int64, [None, 2], name='coordinate')
        # [ΣK]
        self.number = tf.placeholder(tf.int64, [None], name='number')

        self.gt_map = tf.placeholder(tf.int64, shape=[cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], 1],name='gt_map')

        self.apollo_8feature = tf.placeholder(tf.float32, shape=[None, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], 8], name='apollo_feature')



        self.vfe_feature = self.vfe_encoder(vfe_size=(32, 128, 32), name="VFE-Encoder", training=True)

        self.apollo_input = tf.concat([self.vfe_feature, self.apollo_8feature], axis=3)

        self.predicted_map = self.apollo_net(self.apollo_input)


    def apollo_net(self, vfe_map_feature):
        # with tf.variable_scope('conv-block0') as scope:
        #     conv0_1 = self.conv2d_relu(vfe_map_feature, num_kernels=24, kernel_size=(1, 1), stride=[1, 1, 1, 1],
        #                                padding='VALID', name='conv0_1')
        #     conv0 = self.conv2d_relu(conv0_1, num_kernels=24, kernel_size=(3, 3), stride=[1, 1, 1, 1],
        #                              padding='SAME', name='conv0')

        with tf.variable_scope('conv-block1') as scope:
            conv1_1 = self.conv2d_relu(vfe_map_feature, num_kernels=48, kernel_size=(3, 3), stride=[1, 2, 2, 1],
                                       padding='SAME', name='conv1_1')
            conv1 = self.conv2d_relu(conv1_1, num_kernels=48, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                     padding='SAME', name='conv1')

        with tf.variable_scope('conv-block2') as scope:
            conv2_1 = self.conv2d_relu(conv1, num_kernels=64, kernel_size=(3, 3), stride=[1, 2, 2, 1],
                                       padding='SAME', name='conv2_1')
            conv2_2 = self.conv2d_relu(conv2_1, num_kernels=64, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                       padding='SAME', name='conv2_2')
            conv2 = self.conv2d_relu(conv2_2, num_kernels=64, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                     padding='SAME', name='conv2')

        with tf.variable_scope('conv-block3') as scope:
            conv3_1 = self.conv2d_relu(conv2, num_kernels=96, kernel_size=(3, 3), stride=[1, 2, 2, 1],
                                       padding='SAME', name='conv3_1')
            conv3_2 = self.conv2d_relu(conv3_1, num_kernels=96, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                       padding='SAME', name='conv3_2')
            conv3 = self.conv2d_relu(conv3_2, num_kernels=96, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                     padding='SAME', name='conv3')

        with tf.variable_scope('conv-block4') as scope:
            conv4_1 = self.conv2d_relu(conv3, num_kernels=128, kernel_size=(3, 3), stride=[1, 2, 2, 1],
                                       padding='SAME', name='conv4_1')
            conv4_2 = self.conv2d_relu(conv4_1, num_kernels=128, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                       padding='SAME', name='conv4_2')
            conv4 = self.conv2d_relu(conv4_2, num_kernels=128, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                     padding='SAME', name='conv4')

        with tf.variable_scope('conv-block5') as scope:
            conv5_1 = self.conv2d_relu(conv4, num_kernels=192, kernel_size=(3, 3), stride=[1, 2, 2, 1],
                                       padding='SAME', name='conv5_1')
            conv5_2 = self.conv2d_relu(conv5_1, num_kernels=192, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                       padding='SAME', name='conv5_2')
            conv5 = self.conv2d_relu(conv5_2, num_kernels=192, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                     padding='SAME', name='conv5')

        with tf.variable_scope('deconv-block4') as scope:
            deconv4_shape0 = tf.shape(conv4)
            deconv4_shape = tf.stack([deconv4_shape0[0], deconv4_shape0[1], deconv4_shape0[2], 128])
            W_t4 = self.weight_variable([4, 4, 128, 192], name="W_t4")
            b_t4 = self.bias_variable([128], name="b_t4")
            deconv4 = self.conv2d_transpose_strided(conv5, W_t4, b_t4, output_shape=deconv4_shape)

            concat4 = self.concat_relu([conv4, deconv4], axis=3, name="concat4")

            conv_deconv4 = self.conv2d_relu(concat4, num_kernels=128, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                            padding='SAME', name='conv_deconv4')

        with tf.variable_scope('deconv-block3') as scope:
            deconv3_shape0 = tf.shape(conv3)
            deconv3_shape = tf.stack([deconv3_shape0[0], deconv3_shape0[1], deconv3_shape0[2], 96])
            W_t3 = self.weight_variable([4, 4, 96, 128], name="W_t3")
            b_t3 = self.bias_variable([96], name="b_t3")
            deconv3 = self.conv2d_transpose_strided(conv_deconv4, W_t3, b_t3, output_shape=deconv3_shape)

            # deconv3 = conv2d_transpose_strided(conv_deconv4, num_kernels=96,name='deconv_3')

            concat3 = self.concat_relu([conv3, deconv3], axis=3, name="concat3")

            conv_deconv3 = self.conv2d_relu(concat3, num_kernels=96, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                            padding='SAME', name='conv_deconv3')

        with tf.variable_scope('deconv-block2') as scope:
            deconv2_shape0 = tf.shape(conv2)
            deconv2_shape = tf.stack([deconv2_shape0[0], deconv2_shape0[1], deconv2_shape0[2], 64])
            W_t2 = self.weight_variable([4, 4, 64, 96], name="W_t2")
            b_t2 = self.bias_variable([64], name="b_t2")
            deconv2 = self.conv2d_transpose_strided(conv_deconv3, W_t2, b_t2, output_shape=deconv2_shape)

            # deconv2 = conv2d_transpose_strided(conv_deconv3, num_kernels=64,name='deconv_2')

            concat2 = self.concat_relu([conv2, deconv2], axis=3, name="concat2")

            conv_deconv2 = self.conv2d_relu(concat2, num_kernels=64, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                            padding='SAME', name='conv_deconv2')

        with tf.variable_scope('deconv-block1') as scope:
            deconv1_shape0 = tf.shape(conv1)
            deconv1_shape = tf.stack([deconv1_shape0[0], deconv1_shape0[1], deconv1_shape0[2], 48])
            W_t1 = self.weight_variable([4, 4, 48, 64], name="W_t1")
            b_t1 = self.bias_variable([48], name="b_t1")
            deconv1 = self.conv2d_transpose_strided(conv_deconv2, W_t1, b_t1, output_shape=deconv1_shape)

            # deconv1 = conv2d_transpose_strided(conv_deconv2, num_kernels=48,name='deconv_1')

            concat1 = self.concat_relu([conv1, deconv1], axis=3, name="concat1")

            conv_deconv4 = self.conv2d_relu(concat1, num_kernels=48, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                            padding='SAME', name='conv_deconv4')

        with tf.variable_scope('deconv-block0') as scope:
            out_size = 2
            deconv0_shape0 = tf.shape(vfe_map_feature)
            deconv0_shape = tf.stack([deconv0_shape0[0], deconv0_shape0[1], deconv0_shape0[2], out_size])
            W_t0 = self.weight_variable([4, 4, out_size, 48], name="W_t0")
            b_t0 = self.bias_variable([out_size], name="b_t0")
            predicted_map = self.conv2d_transpose_strided(conv_deconv4, W_t0, b_t0, output_shape=deconv0_shape)
        # v2 = tf.constant(0.5)
        # scores = predict[:, :, :, :2]
        # # sigmoid = tf.nn.sigmoid(scores, name='scores')
        # probability = tf.nn.sigmoid(scores[:, :, :, 0], name='category')
        # confidence = tf.nn.sigmoid(scores[:, :, :, 1], name='confidence')
        # predicted_map = tf.cast(tf.greater(probability, v2), dtype=tf.int64, name='prediction')
        return predicted_map