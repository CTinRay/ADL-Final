import pdb
import math
import tensorflow as tf


EPSILON = 1e-5


def conv_capsule(inputs, activation, kernel_size, stride, channels_out,
                 routing_iters=3):
    """Build capsule convolution layer.

    Args:
        inputs (tensor): Pose of lower layer.
            Shape: [batch, input_height, input_width, channels_in,
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

    batch_size = tf.shape(inputs)[0]
    input_height = int(inputs.shape[1])
    input_width = int(inputs.shape[2])
    channels_in = int(inputs.shape[3])
    pose_shape = (int(inputs.shape[4]), int(inputs.shape[5]))
    output_height = (input_height - kernel_size) // stride + 1
    output_width = (input_width - kernel_size) // stride + 1

    # flattern inputs to 4d shape
    # so we can use convolution function for image.
    inputs = tf.reshape(
        inputs,
        [batch_size,
         input_height,
         input_width,
         channels_in * pose_shape[0] * pose_shape[1]])

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
        kernel_size ** 2 * channels_in
        * pose_shape[0] * pose_shape[1]])

    # seperate out dimension for poses matrices, so we can do matrix operation.
    conv_poses = tf.reshape(
        conv_poses,
        [batch_size,
         output_height,
         output_width,
         kernel_size ** 2 * channels_in,
         pose_shape[0],
         pose_shape[1]])

    # repeat poses for each output channels
    conv_poses = tf.tile(conv_poses, [1, 1, 1,
                                      channels_out,
                                      1, 1])
    # conv_poses.shape == [
    #     batch,
    #     output_height,
    #     output_width,
    #     channels_out * kernel_size ** 2 * channels_in,
    #     pose_shape[0],
    #     pose_shape[1]]

    # weights of transform matrices
    transform_matrices = tf.get_variable(
        'transform_matrics',
        shape=[channels_out,
               kernel_size ** 2,
               channels_in,
               pose_shape[1],
               pose_shape[1]],
        initializer=tf.truncated_normal_initializer())
    transform_matrices = tf.reshape(
        transform_matrices,
        [1,  # batch_size
         1,
         1,
         channels_out * kernel_size ** 2 * channels_in,
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
    # tiled_transform_matrics.shape[:-2] == conv_poses.shape[:-2]

    # so we can do matrix transformation
    conv_votes = tf.matmul(tiled_transform_matrics, conv_poses)

    # collect activation convolved by upper layer capsules
    conv_active = tf.extract_image_patches(
        activation,
        [1, kernel_size, kernel_size, 1],
        [1, stride, stride, 1],
        [1, 1, 1, 1],
        'VALID')
    # conv_poses.shape == [
    #     batch,
    #     output_height,
    #     output_width,
    #     kernel_size ** 2 * channels_in]

    # start doing EM routing
    conv_votes = tf.reshape(
        conv_votes,
        [batch_size,
         output_height,
         output_width,
         channels_out,
         kernel_size ** 2,
         channels_in,
         pose_shape[0] * pose_shape[1]])
    conv_active = tf.reshape(
        conv_active,
        [batch_size,
         output_height,
         output_width,
         kernel_size ** 2,
         channels_in])

    with tf.variable_scope('em_routing', reuse=tf.AUTO_REUSE):
        output_poses, output_actives = conv_em_routing(
            conv_active, conv_votes, stride, routing_iters)
    # conv_votes = conv_votes * tf.expand_dims(tf.expand_dims(conv_active, 3), -1)
    # output_poses = tf.reduce_mean(conv_votes, [-2, -3])
    output_poses = tf.reshape(
        output_poses,
        [batch_size,
         output_height,
         output_width,
         channels_out,
         pose_shape[0],
         pose_shape[1]])

    # output_actives = tf.reduce_mean(output_poses ** 2, [-1, -2])
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
    r = tf.ones([tf.shape(conv_votes)[0],    # batch
                 int(conv_votes.shape[1]),   # output_height
                 int(conv_votes.shape[2]),   # output_width
                 int(conv_votes.shape[3]),   # channels_out
                 int(conv_votes.shape[4]),   # kernel_size ** 2
                 int(conv_votes.shape[5])],  # channels_in
                name='r_init')
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
                    higher_indices[i * stride + ki][j * stride + kj].append(
                        (i, j, ki * kernel_size + kj))

    # the max number of upper units that convolve a lower unit
    max_convolved = max([max(map(len, arr)) for arr in higher_indices])

    # keep only the batch dimension so we can use tf.gather easily
    r_flattern = tf.reshape(r, [batch_size,
                                r_height * r_width
                                * channels_out
                                * kernel_size ** 2
                                * channels_in])

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
    conv_r_renormed = r / (tf.expand_dims(conv_r_sum, 3) + EPSILON)
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
                            / (2 * tf.expand_dims(tf.expand_dims(s, 4), 5)
                               + EPSILON),
                            -1)
    assert p_exp.shape[1:] == [
        v.shape[1],                 # output_height
        v.shape[2],                 # output_width
        v.shape[3],                 # channels_out
        v.shape[4],                 # kernel_size ** 2
        v.shape[5]]                 # channels_in

    p_denominator = tf.expand_dims(
        tf.expand_dims(tf.reduce_prod(2 * math.pi * s ** 2, -1),
                       -1),
        -1)

    p = tf.exp(p_exp) / (p_denominator + EPSILON)
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
    assert r_prime.shape[1:] == r.shape[1:]

    sum_rv = tf.reduce_sum(tf.expand_dims(r_prime, -1) * v, axis=[-2, -3])
    assert sum_rv.shape[1:] == [
        v.shape[1],   # output_height
        v.shape[2],   # output_width
        v.shape[3],   # channels_out
        v.shape[-1]]  # pose_height * pose_width

    sum_r = tf.expand_dims(
        tf.reduce_sum(tf.reduce_sum(r_prime, 5), 4), -1)
    assert sum_r.shape[1:] == [v.shape[1],   # output_height
                               v.shape[2],   # output_width
                               v.shape[3],   # channels_out
                               1]            # for broadcast

    m = tf.div(sum_rv, sum_r + EPSILON, name='m')
    assert m.shape[1:] == sum_rv.shape[1:]

    r_square_v_minus_m = \
        tf.expand_dims(r_prime, -1) \
        * (v - (tf.expand_dims(tf.expand_dims(m, 4), 5))) ** 2
    sum_r_square_v_minus_m = tf.reduce_sum(
        tf.reduce_sum(r_square_v_minus_m, 5), 4)
    assert sum_r_square_v_minus_m.shape[1:] == m.shape[1:]

    square_s = sum_r_square_v_minus_m / (sum_r + EPSILON)
    s = tf.sqrt(square_s, name='s')
    assert s.shape[1:] == m.shape[1:]

    beta_v = tf.get_variable('beta_v', [1])

    # with tf.control_dependencies([tf.Assert(tf.reduce_min(s) > 0, [s])]):
    cost = (beta_v + tf.log(s + EPSILON)) * sum_r

    beta_a = tf.get_variable('beta_a', [1])
    a_prime = tf.sigmoid(inv_tempt *
                         (beta_a - tf.reduce_sum(cost, -1)))
    assert a_prime.shape[1:] == [v.shape[1],   # output_height
                                 v.shape[2],   # output_width
                                 v.shape[3]]   # channels_out

    return m, s, a_prime


def class_capsule(inputs, activation, n_classes,
                  routing_iters=1):
    """Build class capsule layer.

    Args:
        inputs (tensor): Pose of lower layer.
            Shape: [batch, input_height, input_width, channels_in,
                    pose_height, pose_width]
        activation (tensor): Activation of lower layer.
            Shape: [batch, input_height, input_width, channels_in]
        n_classes (int): Number of output classes.
        routing_iters (int): Number of routing iterations,

    Returns:
        pose (tensor): Output pose tensor.
            Shape: [batch, n_classes, pose_height, pose_width]
        activation (tensor): Output activation tensor.
            Shape: [batch, n_classes]
    """
    batch_size = tf.shape(inputs)[0]
    input_height = int(inputs.shape[1])
    input_width = int(inputs.shape[2])
    channels_in = int(inputs.shape[3])
    pose_shape = int(inputs.shape[4]), int(inputs.shape[5])

    # copy pose of lower level capsules n_classes times
    poses = tf.tile(tf.expand_dims(inputs, axis=1),
                    [1, n_classes, 1, 1, 1, 1, 1])
    assert poses.shape[1:] == [n_classes, input_height, input_width,
                               channels_in, pose_shape[0], pose_shape[1]]

    # weights of transform matrices
    transform_matrices = tf.get_variable(
        'transform_matrics',
        shape=[n_classes,
               channels_in,
               pose_shape[0],
               pose_shape[0]],
        initializer=tf.truncated_normal_initializer())

    # matric transformation
    tiled_transform_matrics = tf.tile(
        tf.reshape(transform_matrices,
                   [1,
                    n_classes,
                    1,
                    1,
                    channels_in,
                    pose_shape[0],
                    pose_shape[0]]),
        [batch_size,
         1,   # n_classes
         input_height,
         input_width,
         1,   # channels_in
         1,   # pose_height
         1])  # pose_height
    assert tiled_transform_matrics.shape[1:] == [n_classes,
                                                 input_height,
                                                 input_width,
                                                 channels_in,
                                                 pose_shape[0],
                                                 pose_shape[0]]

    # do matrix transformation
    votes = tf.matmul(tiled_transform_matrics, poses,
                      name='votes')
    assert votes.shape[1:] == [n_classes,
                               input_height,
                               input_width,
                               channels_in,
                               pose_shape[0],
                               pose_shape[1]]

    # reshape as if lower layer is convolved to 1x1
    votes = tf.reshape(votes,
                       [batch_size,
                        1, 1,
                        n_classes,
                        input_height * input_width,
                        channels_in,
                        pose_shape[0] * pose_shape[1]])
    activation = tf.reshape(activation,
                            [batch_size,
                             1, 1,
                             input_height * input_width,
                             channels_in])

    # do EM-routing
    with tf.variable_scope('em_routing', reuse=tf.AUTO_REUSE):
        output_poses, output_actives = conv_em_routing(
            activation, votes, 1, routing_iters)

    # flattern results from 2d to 1d
    output_poses = tf.reshape(output_poses, [batch_size,
                                             n_classes,
                                             pose_shape[0],
                                             pose_shape[1]],
                              name='output_poses')
    output_actives = tf.reshape(output_actives, [batch_size, n_classes],
                                name='output_actives')
    return output_poses, output_actives
