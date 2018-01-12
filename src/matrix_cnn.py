import tensorflow as tf


def matrix_cnn(inputs, kernel_size, stride):
    """Build capsule convolution layer.

    Args:
        inputs (tensor): Pose of lower layer.
            Shape: [batch, input_height, input_width,
                    pose_height, pose_width]
        kernel_size (int): Size of kernel.
        stride (int): Size of stride.
        channels_out (int): Number of output channel.

    Returns:
        pose (tensor): Output pose tensor.
            Shape: [batch, output_height, output_width,
                    pose_height, pose_width]
    """

    batch_size = tf.shape(inputs)[0]
    input_height = int(inputs.shape[1])
    input_width = int(inputs.shape[2])
    pose_shape = (int(inputs.shape[3]), int(inputs.shape[4]))
    output_height = (input_height - kernel_size) // stride + 1
    output_width = (input_width - kernel_size) // stride + 1

    # flattern inputs to 4d shape
    # so we can use convolution function for image.
    inputs = tf.reshape(
        inputs,
        [batch_size,
         input_height,
         input_width,
         pose_shape[0] * pose_shape[1]])

    # collect pose matrices convolved by upper layer capsules
    conv_poses = tf.extract_image_patches(
        inputs,
        [1, kernel_size, kernel_size, 1],
        [1, stride, stride, 1],
        [1, 1, 1, 1],
        'VALID')
    conv_poses.set_shape([
        None,
        output_height,
        output_width,
        kernel_size ** 2
        * pose_shape[0] * pose_shape[1]])

    # seperate out dimension for poses matrices, so we can do matrix operation.
    conv_poses = tf.reshape(
        conv_poses,
        [batch_size,
         output_height,
         output_width,
         kernel_size ** 2,
         pose_shape[0],
         pose_shape[1]])

    # weights of transform matrices
    transform_matrices = tf.get_variable(
        'transform_matrics',
        shape=[kernel_size ** 2,
               pose_shape[1],
               pose_shape[1]],
        initializer=tf.truncated_normal_initializer())
    transform_matrices = tf.reshape(
        transform_matrices,
        [1,  # batch_size
         1,
         1,
         kernel_size ** 2,
         pose_shape[1],
         pose_shape[1]])

    # matric transformation
    tiled_transform_matrics = tf.tile(
        transform_matrices,
        [batch_size,
         output_height,
         output_width,
         1, 1, 1])

    # now the shape of transform matrices should be same as conv_poses
    assert tiled_transform_matrics.shape[1:-2] == conv_poses.shape[1:-2]

    # so we can do matrix transformation
    conv_votes = tf.matmul(tiled_transform_matrics, conv_poses)

    conv_votes = tf.reshape(
        conv_votes,
        [batch_size,
         output_height,
         output_width,
         kernel_size ** 2,
         pose_shape[0],
         pose_shape[1]])

    output_poses = tf.reduce_mean(conv_votes, [-3])

    assert output_poses.shape[1:] == [output_height,
                                      output_width,
                                      pose_shape[0],
                                      pose_shape[1]]

    return output_poses
