import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import midi
import os

lowerBound = 24
upperBound = 102
span = upperBound-lowerBound

def midiToNoteStateMatrix(midifile, squash=True, span=span):
    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    time = 0

    state = [[0,0] for x in range(span)]
    statematrix.append(state)
    condition = True
    while condition:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            statematrix.append(state)
        for i in range(len(timeleft)): #For each track
            if not condition:
                break
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = [0, 0]
                        else:
                            state[evt.pitch-lowerBound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        out =  statematrix
                        condition = False
                        break
                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    S = np.array(statematrix)
    statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
    statematrix = np.asarray(statematrix).tolist()
    return statematrix

def noteStateMatrixToMidi(statematrix, name="example", span=span):
    statematrix = np.array(statematrix)
    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    span = upperBound-lowerBound
    tickscale = 55

    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.midi".format(name), pattern)


def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs


songs = get_songs('bach')  # These songs have already been converted from midi to msgpack
print("{0} songs processed".format(len(songs)))
###################################################

# HyperParameters
num_timesteps = 15  # This is the number of timesteps that we will create at a time
n_visible = 2 * note_range * num_timesteps  # This is the size of the visible layer.
n_hidden = 200  # This is the size of the hidden layer

num_epochs = 200  # The number of training epochs that we are going to run.
# For each epoch we go through the entire data set.
batch_size = 100  # The number of training examples that we are going to send through the RBM at a time.
lr = tf.constant(0.005, tf.float32)  # The learning rate of our model

# Variables:
# Next, let's look at the variables we're going to use:
# The placeholder variable that holds our data
x = tf.placeholder(tf.float32, [None, n_visible], name="x")
# The weight matrix that stores the edge weights
W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W")
# The bias vector for the hidden layer
bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="bh"))
# The bias vector for the visible layer
bv = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="bv"))


# Helper functions.
# This function lets us easily sample from a vector of probabilities
def sample(probs):
    # Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


# This function runs the gibbs chain. We will call this function in two places:
#    - When we define the training update step
#    - When we sample our music segments from the trained RBM
def gibbs_sample(k):
    # Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        # Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))  # Propagate the visible values to sample the hidden values
        xk = sample(
            tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))  # Propagate the hidden values to sample the visible values
        return count + 1, k, xk

    # Run gibbs steps for k iterations
    ct = tf.constant(0)  # counter
    [_, _, x_sample] = tf.while_loop(lambda count, num_iter, *args: count < num_iter,
                                                   gibbs_step, [ct, tf.constant(k), x])
    # This is not strictly necessary in this implementation,
    # but if you want to adapt this code to use one of TensorFlow's
    # optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample


# Training Update Code
# Now we implement the contrastive divergence algorithm.
# First, we get the samples of x and h from the probability distribution
# The sample of x
x_sample = gibbs_sample(1)
# The sample of the hidden nodes, starting from the visible state of x
h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
# The sample of the hidden nodes, starting from the visible state of x_sample
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

# Next, we update the values of W, bh, and bv,
# based on the difference between the samples that we drew and the original values
size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder = tf.multiply(lr / size_bt,
                      tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
bh_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
# When we do sess.run(updt), TensorFlow will run all 3 update steps
updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]

# Run the graph!
# Now it's time to start a session and run the graph!

with tf.Session() as sess:
    # First, we train the model
    # initialize the variables of the model
    init = tf.global_variables_initializer()
    sess.run(init)
    # Run through all of the training data num_epochs times
    for epoch in tqdm(range(num_epochs)):
        for song in songs:
            # The songs are stored in a time x notes format. The size of each song is timesteps_in_song x 2*note_range
            # Here we reshape the songs so that each training example
            # is a vector with num_timesteps x 2*note_range elements
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0] // num_timesteps) * num_timesteps)]
            song = np.reshape(song, [song.shape[0] // num_timesteps, song.shape[1] * num_timesteps])
            # Train the RBM on batch_size examples at a time
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i + batch_size]
                sess.run(updt, feed_dict={x: tr_x})

    # Now the model is fully trained, so let's make some music!
    # Run a gibbs chain where the visible nodes are initialized to 0
    sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((10, n_visible))})
    for i in range(sample.shape[0]):
        if not any(sample[i, :]):
            continue
        # Here we reshape the vector to be time x notes, and then save the vector as a midi file
        S = np.reshape(sample[i, :], (num_timesteps, 2 * note_range))
        noteStateMatrixToMidi(S, "generated/composition_{}".format(i))
