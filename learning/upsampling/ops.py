import tensorflow.compat.v1 as tf

def conv2d(input, output_chn, kernel_size, stride=1, dilation=(1,1), use_bias=True, name='conv'):
    return tf.layers.conv2d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            dilation_rate=dilation,padding='same', data_format='channels_last',
                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            use_bias=use_bias, name=name)


def conv_relu(input, output_chn, kernel_size, stride=1, dilation=(1,1), use_bias=True, name='conv'):
    with tf.variable_scope(name):
        conv = conv2d(input, output_chn, kernel_size, stride, dilation, use_bias,  name)
        relu = tf.nn.relu(conv)

    return relu

def conv_lrelu(input, output_chn, kernel_size, stride=1, dilation=(1,1), use_bias=True, name='conv'):
    with tf.variable_scope(name):
        conv = conv2d(input, output_chn, kernel_size, stride, dilation, use_bias,  name)
        relu = tf.nn.leaky_relu(conv,0.1)

    return relu

def amount_block(input, output_chn, kernel_size, name='conv'):
    with tf.variable_scope(name):
        conv1 = conv_lrelu(input, output_chn=output_chn, kernel_size=kernel_size, dilation=(1,1), name = 'conv1')
        conv2 = conv_lrelu(conv1, output_chn=output_chn, kernel_size=kernel_size, dilation=(2,2), name = 'conv2')
        conv3 = conv_lrelu(conv2, output_chn=output_chn, kernel_size=kernel_size, dilation=(4,4), name = 'conv3')
        conv4 = conv_lrelu(conv3, output_chn=output_chn, kernel_size=kernel_size, dilation=(8,8), name = 'conv4')
        
        output_conv = conv2d(conv4, output_chn=1, kernel_size=kernel_size, name = 'output_conv')
        output_concat = tf.concat([input,conv4], axis=3, name='output_concat')

        return output_conv, output_concat

####################
# box filter and guided filter
####################

def diff_x(input, r):
    assert input.shape.ndims == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = tf.concat([left, middle, right], axis=2)

    return output


def diff_y(input, r):
    assert input.shape.ndims == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = tf.concat([left, middle, right], axis=3)

    return output


def box_filter_op(x, r):
    assert x.shape.ndims == 4

    return diff_y(tf.cumsum(diff_x(tf.cumsum(x, axis=2), r), axis=3), r)

def box_filter(x, r, nhwc=True):
    assert x.shape.ndims == 4

    if nhwc:
        x = tf.transpose(x, [0, 3, 1, 2])

    x_shape = tf.shape(x)
    x = tf.identity(x)

    N = box_filter_op(tf.ones((1, 1, x_shape[2], x_shape[3]), dtype=x.dtype), r)

    mean_x = box_filter_op(x, r) / N

    if nhwc:
        mean_x = tf.transpose(mean_x, [0, 2, 3, 1])

    return mean_x

def guided_filter(x, y, r=4, eps=1e-2, nhwc=True):
    assert x.shape.ndims == 4 and y.shape.ndims == 4

    # data format
    if nhwc:
        x = tf.transpose(x, [0, 3, 1, 2])
        y = tf.transpose(y, [0, 3, 1, 2])

    # shape check
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)

    assets = [tf.assert_equal(   x_shape[0],  y_shape[0]),
              tf.assert_equal(  x_shape[2:], y_shape[2:]),
              tf.assert_greater(x_shape[2:],   2 * r + 1),
              tf.Assert(tf.logical_or(tf.equal(x_shape[1], 1),
                                      tf.equal(x_shape[1], y_shape[1])), [x_shape, y_shape])]

    with tf.control_dependencies(assets):
        x = tf.identity(x)

    # N
    N = box_filter_op(tf.ones((1, 1, x_shape[2], x_shape[3]), dtype=x.dtype), r)

    # mean_x
    mean_x = box_filter_op(x, r) / N
    mean_mean_x = box_filter_op(mean_x, r) / N
    # mean_y
    mean_y = box_filter_op(y, r) / N
    mean_mean_y = box_filter_op(mean_y, r) / N
    # cov_xy
    cov_xy = box_filter_op(x * y, r) / N - mean_x * mean_y
    # var_x
    var_x  = box_filter_op(x * x, r) / N - mean_x * mean_x

    # A
    A = cov_xy / (var_x + eps)
    # b
    b = mean_y - A * mean_x

    mean_A = box_filter_op(A, r) / N
    mean_b = box_filter_op(b, r) / N

    output = mean_A * x + mean_b

    if nhwc:
        output = tf.transpose(output, [0, 2, 3, 1])

    return output
