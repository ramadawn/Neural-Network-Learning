import tensorflow as tf

x = tf.cast(tf.constant(15), tf.float64)
y = tf.cast(tf.constant(5), tf.float64)
c = tf.cast(tf.constant(1), tf.float64)

##x = tf.cast(x , tf.float64)
##y = tf.cast(y , tf.float64)
##c = tf.cast(c , tf.float64)

z = tf.subtract(tf.divide(x,y), c)

with tf.Session() as session:
    output = session.run(z)
    print(output)
