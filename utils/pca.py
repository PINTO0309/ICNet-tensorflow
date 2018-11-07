import tensorflow as tf

def pca(input, out_dims=3):
    """pca
    Args:
      input: [n, h, w, c]
      out_dims: output channels
    Outputs:
      Tensor [n, h, w, out_dims]
    """

    # im2col
    input_shape = tf.shape(input)
    n = input_shape[0]
    h = input_shape[1]
    w = input_shape[2]
    c = input_shape[3]
    feature_vec = tf.reshape(input, [n*h*w, c])

    # svd
    s, u, _ = tf.svd(feature_vec)
    sigma = tf.diag(s)

    # select principal component
    sigma = tf.slice(sigma, [0,0], [c, out_dims])

    feature_pca = tf.matmul(u, sigma)
    feature_pca = tf.reshape(feature_pca, [n, h, w, out_dims])
    return feature_pca
    