import tensorflow as tf


def capsule_conv(inputs, activation, kernel_size, stride, channels_out,
                 routing_iters=3):
    """Build capsule convolution layer.

    Args:
        inputs (tensor): Pose of lower layer.
            Shape: [input_height, input_width, n_channels_in, pose_height,
            pose_width]
        activation (tensor): Activation of lower layer.
            Shape: [input_height, input_width, n_channels_in, 1]
        kernel_size (int): Size of kernel.
        stride (int): Size of stride.
        channels_out (int): Number of output channel.
        routing_iters (int): Number of routing iterations,

    Returns:
        pose (tensor): Output pose tensor. Shape: [output_height, output_width,
            n_channels_out * pose_height * pose_width]
        activation (tensor): Output activation tensor.
    """

    channels_in = inputs.shape[-3]
    pose_shape = inputs.shape[-2:]

    # flattern inputs to shape
    # [batch, input_height, input_width,
    #  channels_in * pose_height * pose_width]
    # so we can use convolution function for image.
    inputs = tf.reshape(
        inputs,
        [inputs.shape[0], inputs.shape[1], inputs.shape[2], -1])

    # collect pose matrices convolved by upper layer
    conv_poses = tf.extract_image_paches(
        inputs,
        [1, kernel_size, kernel_size, 1],
        [1, stride, stride, 1],
        [1, 1, 1, 1],
        'VALID')

    assert conv_poses.shape == [
        inputs.shape[0],  # batch
        (inputs.shape[1] - kernel_size) // stride + 1,  # output height
        (inputs.shape[2] - kernel_size) // stride + 1,  # output width
        kernel_size ** 2 * channels_in * pose_shape[0] * pose_shape[1]]

    # seperate out dimension for poses matrices, so we can do matrix operation.
    conv_poses = tf.reshape(
        conv_poses,
        [conv_poses.shape[0],
         conv_poses.shape[1],
         kernel_size ** 2 * channels_in,
         pose_shape[0],
         pose_shape[1]])

    # repeat poses for each output channels
    conv_poses = tf.cat([inputs] * channels_out, 2)

    assert conv_poses.shape == [
        inputs.shape[0],  # batch
        (inputs.shape[1] - kernel_size) // stride + 1,  # output height
        (inputs.shape[2] - kernel_size) // stride + 1,  # output width
        channels_out * kernel_size ** 2 * channels_in,
        pose_shape[0],
        pose_shape[1]]

    # weights of transform matrices
    transform_matrices = tf.get_variable(
        'transform_matrices',
        shape=[channels_out * kernel_size ** 2 * channels_in,
               pose_shape[1],
               pose_shape[1]])

    # matric transformation
    tiled_transform_matrics = tf.tile(
        tf.expand_dims(
            tf.expand_dims(tf.expand_dims(transform_matrices, 1), 1)),
        [1,  # batch dim remains unchanged
         conv_poses.shape[0],  # output height
         conv_poses.shape[1],  # output width
         1, 1, 1])

    # now the shape of transform matrices should be same as conv_poses
    assert tiled_transform_matrics.shape == conv_poses

    # so we can do matrix transformation
    conv_votes = tf.matmul(tiled_transform_matrics, conv_poses)

    # start doing EM routing
    output_poses, output_actives = em_routing_conv(
        activation, conv_votes, routing_iters)

    return output_poses, output_actives


def em_routing_conv(activation, conv_votes, routing_iters):
    """EM routing algorithm of CapsuleEM.

    Args:
        activation (tensor): Activation of lower capsules.
            Shape: [batch, input_height, input_height, 1]
        conv_votes (tensor): Votes of lower capsuls.
            Shape: [batch, output_height, output_width,
                    channels_out * kernel_size ** 2 * channels_in,
                    pose_height, pose_width]
        routing_iters (int): Number of routing iterations to do.
    """
    def e_step(m, s, a, v):
        pass

    def m_step(r, a_prime, v):
        pass

    # initialze r [TODO]
    r = None

    # start EM loop
    for i in range(routing_iters):
        m, s, a = m_step(r, activation, conv_votes)
        r = e_step(m, s, a, conv_votes)

    return m, a
