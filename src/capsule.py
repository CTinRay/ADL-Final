import pdb
import math
import tensorflow as tf


def capsule_conv(inputs, activation, kernel_size, stride, channels_out,
                 routing_iters=3):
    """Build capsule convolution layer.

    Args:
        inputs (tensor): Pose of lower layer.
            Shape: [batch, input_height, input_width, n_channels_in,
                    pose_height, pose_width]
        activation (tensor): Activation of lower layer.
            Shape: [batch, input_height, input_width, channels_in]
        kernel_size (int): Size of kernel.
        stride (int): Size of stride.
        channels_out (int): Number of output channel.
        routing_iters (int): Number of routing iterations,

    Returns:
        pose (tensor): Output pose tensor.
            Shape: [batch, output_height, output_width, channels_out,
                    pose_height, pose_width]
        activation (tensor): Output activation tensor.
            Shape: [batch, output_height, output_width, channels_out]
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

    # collect pose matrices convolved by upper layer capsules
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

    # collect activation convolved by upper layer capsules
    conv_active = tf.extract_image_paches(
        activation,
        [1, kernel_size, kernel_size, 1],
        [1, stride, stride, 1],
        [1, 1, 1, 1],
        'VALID')
    assert conv_poses.shape == [
        conv_votes.shape[0],  # batch
        conv_votes.shape[1],  # output height
        conv_votes.shape[2],  # output width
        kernel_size ** 2 * channels_in]

    # start doing EM routing
    conv_votes = tf.reshape(
        conv_votes,
        [conv_votes.shape[0],  # batch
         conv_votes.shape[1],  # output_height
         conv_votes.shape[2],  # output_width
         channels_out * kernel_size ** 2,
         channels_in,
         pose_shape[0] * pose_shape[1]])
    conv_active = tf.reshape(
        conv_active,
        [conv_active.shape[0],  # batch
         conv_active.shape[1],  # output_height
         conv_active.shape[2],  # output_width
         kernel_size ** 2,
         channels_in])

    output_poses, output_actives = conv_em_routing(
        conv_active, conv_votes, stride, routing_iters)

    return output_poses, output_actives


def conv_em_routing(activation, conv_votes, stride, routing_iters):
    """EM routing algorithm of CapsuleEM.

    Args:
        activation (tensor): Activation of lower capsules.
            Shape: [batch, output_height, output_width,
                    kernel_size ** 2, channels_in]
        conv_votes (tensor): Votes of lower capsuls.
            Shape: [batch, output_height, output_width, channels_out,
                    kernel_size ** 2, channels_in,
                    pose_height * pose_width]
        stride (int): Stride of the convolution layer before routing.
        routing_iters (int): Number of routing iterations to do.

    Returns:
        m (tensor): Pose of capsules.
            Shape: [batch, output_height, output_width, channels_out,
                    pose_height, pose_width]
        a_prime (tensor): Activation of capsules.
            Shape: [batch, output_height, output_width, channels_out]
    """
    # initialze r
    r = tf.ones(conv_votes.shape[:3] + [conv_votes.shape[4]])
    r = _renorm_r(r, stride)

    # start EM loop
    # [TODO] Schedule inv_tempt (lambda).
    inv_tempt = 1
    for i in range(routing_iters):
        m, s, a_prime = _conv_m_step(r, activation, conv_votes, inv_tempt)
        r = _conv_e_step(m, s, a_prime, conv_votes, stride)

    return m, a_prime


def _renorm_r(r, stride):
    """Renorm r for each capsule in lower layer, its contribution to upper
    layer capsules sum to 1.

    Args:
        r (tensor): Expected portion of lower capsule that belong to
            the upper capsule.
            Shape: [batch, output_height, output_width, channels_out,
                    kernel_size ** 2, channels_in]
        stride (int): Stride of the convolution layer before routing.
    """
    kernel_size = math.sqrt(int(r.shape[-2]))
    assert kernel_size.is_integer()
    kernel_size = int(kernel_size)

    batch_size = tf.shape(r)[0]
    r_height = int(r.shape[1])
    r_width = int(r.shape[2])
    origin_h = r_height * stride + (kernel_size - 1)
    origin_w = r_width * stride + (kernel_size - 1)

    channels_out = int(r.shape[3])
    channels_in = int(r.shape[-1])

    # collect indices of higher level units that convolve lower level
    # unit at i, j.
    higher_indices = [[[] for w in range(origin_w)]
                      for h in range(origin_h)]
    for i in range(r_height):
        for j in range(r_width):
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    higher_indices[i + ki][j + kj].append(
                        (i, j, ki * kernel_size + kj))

    # the max number of upper units that convolve a lower unit
    max_convolved = max([max(map(len, arr)) for arr in higher_indices])

    # keep only the batch dimension so we can use tf.gather easily
    r_flattern = tf.reshape(r, [batch_size,
                                r_height * r_width
                                * channels_out
                                * kernel_size ** 2
                                * channels_out])

    # pad r with 0 so zero_index will point to 0
    r_flattern = tf.concat([r_flattern, tf.zeros([batch_size, 1])], axis=-1)

    # index that point to 0
    zero_index = int(r_flattern.shape[-1]) - 1

    sum_r_indices = []
    for i in range(origin_h):
        for j in range(origin_w):
            for cout in range(channels_out):
                for k in range(max_convolved):
                    for cin in range(channels_in):
                        if k < len(higher_indices[i][j]):
                            higher_i, higher_j, k_shift \
                                = higher_indices[i][j][k]
                            index = ((((higher_i * r_height + higher_j)
                                       * channels_out + cout)
                                      * kernel_size ** 2 + k_shift)
                                     * channels_in + cin)
                            sum_r_indices.append(index)
                        else:
                            # for border units that are convolved less times
                            # append padding index that points to 0
                            sum_r_indices.append(zero_index)

    # gather r that is contributed from lower layer unit i, j
    r_gather = tf.gather(r_flattern, sum_r_indices, axis=-1)
    r_gather = tf.reshape(r_gather, [-1,
                                     origin_h,
                                     origin_w,
                                     channels_out * max_convolved,
                                     channels_in])

    # summation of r
    r_sum = tf.reduce_sum(r_gather, -2)

    # collect r_sum
    conv_r_sum = tf.extract_image_patches(
        r_sum,
        [1, kernel_size, kernel_size, 1],
        [1, stride, stride, 1],
        [1, 1, 1, 1],
        'VALID')

    # calculate summation of r
    conv_r_sum = tf.reshape(
        conv_r_sum,
        [batch_size,
         r_height,
         r_width,
         kernel_size ** 2,
         channels_in])

    # renorm r by divide original r with conv_r_sum
    conv_r_renormed = r / tf.expand_dims(conv_r_sum, 3)
    assert conv_r_renormed.shape[1:] == r.shape[1:]

    return conv_r_renormed


def _conv_e_step(m, s, a_prime, v, stride):
    """E-step of the EM algorithm.
    Note that only VALID padding is supported (when renorming r).

    Args:
        m (tensor): Pose of capsules.
            Shape: [batch, output_height, output_width, channels_out,
                    pose_height, pose_width]
        s (tensor): Standard deviation of capsules.
            Shape: [batch, output_height, output_width, channels_out,
                    pose_height * pose_width]
        a_prime (tensor): Activation of capsules.
            Shape: [batch, output_height, output_width, channels_out]
        v (tensor): Votes of lower capsuls.
            Shape: [batch, output_height, output_width, channels_out,
                    kernel_size ** 2, channels_in,
                    pose_height * pose_width]
        stride (int): Stride of the convolution layer before routing.

    Returns:
        r (tensor): Expected portion of lower capsule that belong to
            the upper capsule.
            Shape: [batch, output_height, output_width, channels_out,
                    kernel_size ** 2, channels_in]
    """
    p_exp = - tf.reduce_sum((v - tf.expand_dims(tf.expand_dims(m, 4), 5)) ** 2
                            / (2 * tf.expand_dims(tf.expand_dims(s, 4), 5)),
                            -1)
    assert p_exp.shape == [
        v.shape[0],                 # batch
        v.shape[1],                 # output_height
        v.shape[2],                 # output_width
        v.shape[3],                 # channels_out
        v.shape[4],                 # kernel_size ** 2
        v.shape[5]]                 # channels_in

    p_denominator = tf.expand_dims(
        tf.expand_dims(tf.reduce_prod(2 * math.pi * s ** 2, -1),
                       -1),
        -1)

    p = tf.exp(p_exp) / p_denominator
    r = p * tf.expand_dims(tf.expand_dims(a_prime, -1), -1)
    r = _renorm_r(r, stride)

    return r


def _conv_m_step(r, a, v, inv_tempt):
    """M-step of the EM algorithm

    Args:
        r (tensor): Expected portion of lower capsule that belong to
            the upper capsule.
            Shape: [batch, output_height, output_width, channels_out,
                    kernel_size ** 2, channels_in]
        a (tensor): Activation of lower capsule.
            Shape: [batch, output_height, output_width,
                    kernel_size ** 2, channels_in]
        v (tensor): Votes of lower capsuls.
            Shape: [batch, output_height, output_width, channels_out,
                    kernel_size ** 2, channels_in,
                    pose_height * pose_width]
        inv_tempt (float): Inverse temperature (lambda).

    Returns:
        m (tensor): Pose of capsules.
            Shape: [batch, output_height, output_width, channels_out,
                    pose_height, pose_width]
        s (tensor): Standard deviation of capsules.
            Shape: [batch, output_height, output_width, channels_out,
                    pose_height * pose_width]
        a_prime (tensor): Activation of capsules.
            Shape: [batch, output_height, output_width, channels_out]
    """
    r_prime = r * tf.expand_dims(a, 3)
    assert r_prime.shape == r.shape

    sum_rv = tf.reduce_sum(
        tf.reduce_sum(tf.expand_dims(r_prime, -1) * v, axis=-2),
        axis=-1)
    assert sum_rv.shape == [
        v.shape[0],                 # batch
        v.shape[1],                 # output_height
        v.shape[2],                 # output_width
        v.shape[3],                 # channels_out
        v.shape[-2] * v.shape[-1]]  # pose_height * pose_width

    sum_r = tf.expand_dims(
        tf.reduce_sum(tf.reduce_sum(r_prime, 5), 4), -1)
    assert sum_r.shape == [v.shape[0],   # batch
                           v.shape[1],   # output_height
                           v.shape[2],   # output_width
                           v.shape[3],   # channels_out
                           1]            # for broadcast

    m = sum_rv / sum_r
    assert m.shape == sum_rv.shape

    r_square_v_minus_m = \
        tf.expand_dims(r_prime, -1) \
        * (v - (tf.expand_dims(tf.expand_dims(m, 4), 5))) ** 2
    sum_r_square_v_minus_m = tf.reduce_sum(
        tf.reduce_sum(r_square_v_minus_m, 5), 4)
    assert sum_r_square_v_minus_m.shape == m.shape

    square_s = sum_r_square_v_minus_m / sum_r
    s = tf.sqrt(square_s)
    assert s.shape == m.shape

    beta_v = tf.get_variable('beta_v')
    cost = (beta_v + tf.log(s)) * sum_r

    beta_a = tf.get_variable('beta_a')
    a_prime = tf.sigmoid(inv_tempt *
                         (beta_a - tf.reduce_sum(tf.reduce_sum(cost, -2), -1)))
    assert a_prime.shape == [v.shape[0],   # batch
                             v.shape[1],   # output_height
                             v.shape[2],   # output_width
                             v.shape[3]]   # channels_out

    return m, s, a_prime
