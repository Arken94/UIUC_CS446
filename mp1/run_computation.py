import tensorflow as tf


def run_computation(inp):
    '''
    This function takes all of the steps necessary to run the provided
    tensor, inp. More specifically:
        -First, create a session
        -Next, initialize variables
        -Then, evaluate the tensor
        -Finally, return the evaluated value

    Args:
       inp(tf.Tensor): an arbitrary Tensor to be evaluated
    Returns:
       The result of the computation (exact type depends on input)

    '''
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    return session.run(inp)
