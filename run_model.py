from __future__ import print_function
import tensorflow as tf
import numpy as np
from PIL import Image
import face_recognition
import os

# Parameters
learning_rate = 0.001
training_iters = 3000
batch_size = 20
display_step = 1

# Network Parameters
n_input = 128*128 # Cropped Image data input (img shape: 128*128 )
n_classes = 10 # Total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

#Cropped Images Directory
RAW_IMG_DIC = './test_images/'
CROPPED_IMG_DIC = './test_images_cropped/'

# Store layers weight & bias
weights = {
    # 5x5 conv, 3 input, 16 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 16])),
    # 5x5 conv, 16 inputs, 32 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([5, 5, 32 ,64])),
    # fully connected, 32*32*96 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([16*16*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def crop_face(raw_img_file):
    """Find all the faces and crop the file in images folder to 128 * 128
    """
    print('---------------------------')
    print('Processing image {}'.format(raw_img_file))
    raw_img = face_recognition.load_image_file(RAW_IMG_DIC + raw_img_file)
    face_locations = face_recognition.face_locations(raw_img)
    print('{} faces found in this image'.format(len(face_locations)))
    count = 0
    for face_loc in face_locations:
        top, right, bottom, left = face_loc
        img_save_name = '{}_{}'.format(str(count), raw_img_file)
        face_img = raw_img[top:bottom, left:right]
        pil_image = Image.fromarray(face_img)
        resized_img = pil_image.resize((128,128))
        resized_img.save(CROPPED_IMG_DIC + img_save_name)
def crop_all():
    raw_img_list = os.listdir(RAW_IMG_DIC)
    for f_name in raw_img_list:
        if '.png' in f_name:
            crop_face(f_name)

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases, dropout):
    #Reshape input picture
    x = tf.reshape(x, shape=[-1, 128, 128, 3])

    #Conv Layer and ReLU #1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print(conv1.shape)

    #Max Pool #1
    mp1 = maxpool2d(conv1, k = 2)

    #Conv Layer and ReLU #2
    conv2 = conv2d(mp1, weights['wc2'], biases['bc2'])
    print(conv2.shape)

    #Max Pool #2
    mp2 = maxpool2d(conv2, k = 2)

    #Conv Layer and ReLU #3
    conv3 = conv2d(mp2, weights['wc3'], biases['bc3'])
    print(conv3.shape)

    #Max Pool #3
    mp3 = maxpool2d(conv3, k = 2)

    #Fully Connected Neural Network #1
    fc1 = tf.reshape(mp3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    #Dropout Layer
    fc1 = tf.nn.dropout(fc1, dropout)

    #Fully Connected Neural Network #2 Output Layer
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out





# Construct model
pred = conv_net(x, weights, biases, keep_prob)
pred_result=tf.argmax(pred, 1)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './model.ckpt')
    crop_all()
    
    xs = []
    fnames = os.listdir(CROPPED_IMG_DIC)
    for fname in fnames:
        img = Image.open(CROPPED_IMG_DIC + fname)
        img_ndarray = np.asarray(img, dtype='float32')
        img_ndarray = np.reshape(img_ndarray, [128, 128, 3])
        xs.append(img_ndarray)
    xs = np.asarray(xs)
    print(xs.shape)

    max_index=sess.run(pred_result, feed_dict={x: xs, keep_prob: 1.})
    for i in range(0, len(fnames)):
        print('File: {} Result: {}'.format(fnames[i], max_index[i] + 1))
