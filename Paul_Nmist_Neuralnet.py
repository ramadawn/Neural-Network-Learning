import tensorflow as tf

"""

input > weight > hidden layer 1 (activation function) > weights > hidden 1 2
(activation fuction) > weights > outputlayer

compare output to intended output > cost function (cross entropy)
optimization function > minimize cost (optimizer)

backpropegation

feed forward + backprop = epoch
"""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 10 classes 0-9

n_nodes_hl1 = 748
n_nodes_hl2 = 748
n_nodes_hl3 = 748
#n_nodes_hl4 = 500

n_classes = 10
batch_size = 100
#squished_x = 784.0
#height x width
x = tf.placeholder('float',[None,784 ]) #flattend 28 x 28
y = tf.placeholder('float')

layerDepth = 5

dataPoints = 784
nodes = 784

def neural_network_model(data):
    hiddenLayers = []
    activationFunctions = []
    currentData = data
    for i in range(layerDepth):
        currentWeights = tf.Variable(tf.random_normal([nodes, nodes]))
        currentBiases = tf.Variable(tf.random_normal([nodes]))
        currentData = tf.nn.relu(tf.add(tf.matmul(currentData, currentWeights), currentBiases))
        
    currentWeights = tf.Variable(tf.random_normal([nodes, n_classes]))
    currentBiases = tf.Variable(tf.random_normal([n_classes]))
    return tf.matmul(currentData, currentWeights) + currentBiases

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #feed forward +backpropigation
    hm_epochs = 10
    
    #opens, runs and then closes a session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #training the network, optimizing weights
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for not_used_variable in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                not_used_variable, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
##                print ("Epoch x = ", epoch_x)
##                print ("Epoch y = ", epoch_y)
##                print ("c = ", c)
                epoch_loss += c
                #print ("Epoch loss = ", epoch_loss)
            print("Epoch ", epoch, "completed out of ", hm_epochs, "loss: ", epoch_loss)
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print("Accuracy: ",accuracy.eval({x:mnist.test.images, y:mnist.test.labels})) 

        #comparing predition to accuracy
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print("Accuracy: ",accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
    
