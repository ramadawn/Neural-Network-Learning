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

n_nodes_hl1 = 784
n_nodes_hl2 = 784
n_nodes_hl3 = 784
#n_nodes_hl4 = 500

n_classes = 10
batch_size = 100
#squished_x = 784.0
#height x width
x = tf.placeholder('float',[None,784 ]) #flattend 28 x 28
y = tf.placeholder('float')

def neural_network_model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    #hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      #'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}   

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    #sum box
    l1 = tf.nn.relu(l1) #activation function rectified linear

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    #sum box
    l2 = tf.nn.relu(l2) #activation function rectified linear

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    #sum box
    l3 = tf.nn.relu(l3) #activation function rectified linear

    #l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']),hidden_4_layer['biases'])
  
    #l4 = tf.nn.relu(l4) #activation function rectified linear 

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #feed forward +backpropigation
    hm_epochs = 10
    save_file = 'hand_write.ckpt'
    saver = tf.train.Saver()
    #opens, runs and then closes a session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #training the network, optimizing weights
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for not_used_variable in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                not_used_variable, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch ", epoch, "completed out of ", hm_epochs, "loss: ", epoch_loss)
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print("Accuracy: ",accuracy.eval({x:mnist.test.images, y:mnist.test.labels})) 
            saver.save(sess, save_file)
            
        #comparing predition to accuracy
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print("Accuracy: ",accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        
train_neural_network(x)
    
