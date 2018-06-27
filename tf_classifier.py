import os
from apt_importers import *
import seaborn as sns
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from pandas import get_dummies

# Reading the main file
dpos = get_dpos('R12_Al-Sc.epos', 'ranges.rrng')

print 'Data Loaded'

dpos = dpos.loc[dpos.ns < 550]
data = dpos.loc[:, ['DC_kV', 'ipp', 'ns', 'pulse_kV', 'element']]

viz = False
if viz is True:
    sns.pairplot(data, hue="element")

X = data.drop(columns='element')
y = get_dummies(data.element)

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

# Converting to float32
X_train = np.array(X_train).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)


num_features = X_test.shape[1]
num_labels = y_test.shape[1]
num_hidden = [5, 3]   # Number of features in a Hidden Layer

learning_rate = 0.005

print 'Data Loaded'

import tensorflow as tf
graph = tf.Graph()


def model(feed_data, w1, b1, w2, b2, w3, b3):

    layer1 = tf.matmul(feed_data, w1) + b1
    relu1 = tf.nn.relu(layer1)
    layer2 = tf.matmul(relu1, w2) + b2
    relu2 = tf.nn.relu(layer2)
    layer3 = tf.matmul(relu2, w3) + b3

    predict = tf.nn.softmax(layer3)

    return layer3, predict


def accuracy(prediction, labels):

    predict = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(predict, tf.float32)) * 100.0


with graph.as_default():
    tf_train_set = tf.constant(X_train)
    tf_train_labels = tf.constant(y_train)
    tf_valid_set = tf.constant(X_test)
    tf_valid_labels = tf.constant(y_test)

    print tf_train_set
    print tf_train_labels

    weights_1 = tf.Variable(tf.truncated_normal([num_features, num_hidden[0]]))
    weights_2 = tf.Variable(tf.truncated_normal([num_hidden[0], num_hidden[1]]))
    weights_3 = tf.Variable(tf.truncated_normal([num_hidden[1], num_labels]))

    bias_1 = tf.Variable(tf.zeros([num_hidden[0]]))
    bias_2 = tf.Variable(tf.zeros([num_hidden[1]]))
    bias_3 = tf.Variable(tf.zeros([num_labels]))

    layer, predict_train = model(tf_train_set, weights_1, bias_1, weights_2, bias_2, weights_3, bias_3)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=layer))
    #tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    training_op = optimizer.minimize(loss=loss)
    tf.summary.scalar('loss', loss)
    _, predict_valid = model(tf_valid_set, weights_1, bias_1, weights_2, bias_2, weights_3, bias_3)
    validation_accuracy = accuracy(predict_valid, tf_valid_labels)
    tf.summary.scalar('Validation_Accuracy', validation_accuracy)
    merge = tf.summary.merge_all()

num_steps = 10000
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    # op to write logs
    train_writer = tf.summary.FileWriter('./logs/train', graph=graph)

    print(loss.eval())
    for step in range(num_steps):
        _, l, acc, summary = session.run([training_op, loss, validation_accuracy, merge])

        # Write logs at every iteration
        train_writer.add_summary(summary=summary, global_step=step)

        if step % 2000 == 0:
            # print(predictions[3:6])
            print('Loss at step %d: %f' % (step, l))
            print('Validation accuracy: %.1f%%' % acc)

    _, l, acc = session.run([training_op, loss, validation_accuracy])
    print('Loss at step %d: %f' % (num_steps, l))
    #print('Training accuracy: %.1f%%' % accuracy(predictions, y_train[:, :]))
    print('Validation accuracy: %.1f%%' % acc)

