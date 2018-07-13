from __future__ import print_function
from builtins import range

__version__ = "1.6"

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import init_ops
try:
    from tensorflow.contrib import nccl
    have_nccl = True
except ImportError:
    have_nccl = False
    print("WARNING: NCCL support not available")

import sys
import os
import time
import math
from collections import defaultdict
import argparse
import numpy as np
import cv2

class DummyScope(object):
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class GPUNetworkBuilder(object):
    def __init__(self,
                 is_training,
                 dtype=tf.float32,
                 activation='RELU',
                 use_batch_norm=True,
                 batch_norm_config = {'decay':   0.9,
                                      'epsilon': 1e-4,
                                      'scale':   True,
                                      'zero_debias_moving_mean': False},
                 use_xla=False):
        self.dtype             = dtype
        self.activation_func   = activation
        self.is_training       = is_training
        self.use_batch_norm    = use_batch_norm
        self.batch_norm_config = batch_norm_config
        self._layer_counts     = defaultdict(lambda: 0)
        if use_xla:
            self.jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
        else:
            self.jit_scope = DummyScope
    def _count_layer(self, layer_type):
        idx  = self._layer_counts[layer_type]
        name = layer_type + str(idx)
        self._layer_counts[layer_type] += 1
        return name
    def _get_variable(self, name, shape, dtype=None,
                      initializer=None, seed=None):
        if dtype is None:
            dtype = self.dtype
        if initializer is None:
            initializer = init_ops.glorot_uniform_initializer(seed=seed)
        elif (isinstance(initializer, float) or
              isinstance(initializer, int)):
            initializer = tf.constant_initializer(float(initializer))
        return tf.get_variable(name, shape, dtype, initializer)
    def _to_nhwc(self, x):
        return tf.transpose(x, [0,2,3,1])
    def _from_nhwc(self, x):
        return tf.transpose(x, [0,3,1,2])
    def _bias(self, input_layer):
        num_outputs = input_layer.get_shape().as_list()[1]
        biases = self._get_variable('biases', [num_outputs], input_layer.dtype,
                                    initializer=0)
        if len(input_layer.get_shape()) == 4:
            return tf.nn.bias_add(input_layer, biases,
                                  data_format='NCHW')
        else:
            return input_layer + biases
    def _batch_norm(self, input_layer, scope):
        return tf.contrib.layers.batch_norm(input_layer,
                                            is_training=self.is_training,
                                            scope=scope,
                                            data_format='NCHW',
                                            fused=True,
                                            **self.batch_norm_config)
    def _bias_or_batch_norm(self, input_layer, scope, use_batch_norm):
        if use_batch_norm is None:
            use_batch_norm = self.use_batch_norm
        if use_batch_norm:
            return self._batch_norm(input_layer, scope)
        else:
            return self._bias(input_layer)
    def input_layer(self, input_layer):
        with self.jit_scope():
            x = self._from_nhwc(input_layer)
            x = tf.cast(x, self.dtype)
            # Rescale and shift to [-1,1]
            x = x * (1./127.5) - 1
        return x
    def conv(self, input_layer, num_filters, filter_size,
             filter_strides=(1,1), padding='SAME',
             activation=None, use_batch_norm=None):
        num_inputs = input_layer.get_shape().as_list()[1]
        kernel_shape = [filter_size[0], filter_size[1],
                        num_inputs, num_filters]
        strides = [1, 1, filter_strides[0], filter_strides[1]]
        with tf.variable_scope(self._count_layer('conv')) as scope:
            kernel = self._get_variable('weights', kernel_shape,
                                        input_layer.dtype)
            if padding == 'SAME_RESNET': # ResNet models require custom padding
                kh, kw = filter_size
                rate = 1
                kernel_size_effective = kh + (kw - 1) * (rate - 1)
                pad_total = kernel_size_effective - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                padding = [[0, 0], [0, 0],
                           [pad_beg, pad_end], [pad_beg, pad_end]]
                input_layer = tf.pad(input_layer, padding)
                padding = 'VALID'
            x = tf.nn.conv2d(input_layer, kernel, strides,
                             padding=padding, data_format='NCHW')
            x = self._bias_or_batch_norm(x, scope, use_batch_norm)
            x = self.activate(x, activation)
            return x
    def deconv(self, input_layer, num_filters, filter_size,
               filter_strides=(2,2), padding='SAME',
               activation=None, use_batch_norm=None):
        num_inputs  = input_layer.get_shape().as_list()[1]
        ih, iw      = input_layer.get_shape().as_list()[2:]
        output_shape = [-1, num_filters,
                        ih*filter_strides[0], iw*filter_strides[1]]
        kernel_shape = [filter_size[0], filter_size[1],
                        num_filters, num_inputs]
        strides = [1, 1, filter_strides[0], filter_strides[1]]
        with tf.variable_scope(self._count_layer('deconv')) as scope:
            kernel = self._get_variable('weights', kernel_shape,
                                        input_layer.dtype)
            x = tf.nn.conv2d_transpose(input_layer, kernel, output_shape,
                                       strides, padding=padding,
                                       data_format='NCHW')
            x = self._bias_or_batch_norm(x, scope, use_batch_norm)
            x = self.activate(x, activation)
            return x
    def activate(self, input_layer, funcname=None):
        if isinstance(funcname, tuple):
            funcname = funcname[0]
            params = funcname[1:]
        if funcname is None:
            funcname = self.activation_func
        if funcname == 'LINEAR':
            return input_layer
        activation_map = {
            'RELU':    tf.nn.relu,
            'RELU6':   tf.nn.relu6,
            'ELU':     tf.nn.elu,
            'SIGMOID': tf.nn.sigmoid,
            'TANH':    tf.nn.tanh,
            'LRELU':   lambda x, name: tf.maximum(params[0]*x, x, name=name)
        }
        return activation_map[funcname](input_layer, name=funcname.lower())
    def pool(self, input_layer, funcname, window_size,
                 window_strides=(2,2),
                 padding='VALID'):
        pool_map = {
            'MAX': tf.nn.max_pool,
            'AVG': tf.nn.avg_pool
        }
        kernel_size    = [1, 1, window_size[0], window_size[1]]
        kernel_strides = [1, 1, window_strides[0], window_strides[1]]
        return pool_map[funcname](input_layer, kernel_size, kernel_strides,
                                  padding, data_format='NCHW',
                                  name=funcname.lower())
    def project(self, input_layer, num_outputs, height, width,
                activation=None):
        with tf.variable_scope(self._count_layer('project')):
            x = self.fully_connected(input_layer, num_outputs*height*width,
                                     activation=activation)
            x = tf.reshape(x, [-1, num_outputs, height, width])
            return x
    def flatten(self, input_layer):
        # Note: This ensures the output order matches that of NHWC networks
        input_layer = self._to_nhwc(input_layer)
        input_shape = input_layer.get_shape().as_list()
        num_inputs  = input_shape[1]*input_shape[2]*input_shape[3]
        return tf.reshape(input_layer, [-1, num_inputs], name='flatten')
    def spatial_avg(self, input_layer):
        return tf.reduce_mean(input_layer, [2, 3], name='spatial_avg')
    def fully_connected(self, input_layer, num_outputs, activation=None):
        num_inputs = input_layer.get_shape().as_list()[1]
        kernel_size = [num_inputs, num_outputs]
        with tf.variable_scope(self._count_layer('fully_connected')):
            kernel = self._get_variable('weights', kernel_size,
                                        input_layer.dtype)
            x = tf.matmul(input_layer, kernel)
            x = self._bias(x)
            x = self.activate(x, activation)
            return x
    def inception_module(self, input_layer, name, cols):
        with tf.name_scope(name):
            col_layers      = []
            col_layer_sizes = []
            for c, col in enumerate(cols):
                col_layers.append([])
                col_layer_sizes.append([])
                x = input_layer
                for l, layer in enumerate(col):
                    ltype, args = layer[0], layer[1:]
                    if   ltype == 'conv': x = self.conv(x, *args)
                    elif ltype == 'pool': x = self.pool(x, *args)
                    elif ltype == 'share':
                        # Share matching layer from previous column
                        x = col_layers[c-1][l]
                    else: raise KeyError("Invalid layer type for " +
                                         "inception module: '%s'" % ltype)
                    col_layers[c].append(x)
            catdim  = 1
            catvals = [layers[-1] for layers in col_layers]
            x = tf.concat(catvals, catdim)
            return x

    def dropout(self, input_layer, keep_prob=0.5):
        if self.is_training:
            dtype = input_layer.dtype
            with tf.variable_scope(self._count_layer('dropout')):
                keep_prob_tensor = tf.constant(keep_prob, dtype=dtype)
                return tf.nn.dropout(input_layer, keep_prob_tensor)
        else:
            return input_layer

def stage(tensors):
    stage_area = data_flow_ops.StagingArea(
        dtypes=[tensor.dtype       for tensor in tensors],
        shapes=[tensor.get_shape() for tensor in tensors])
    put_op      = stage_area.put(tensors)
    get_tensors = stage_area.get()

    get_tensors = [tf.reshape(gt, t.get_shape())
                   for (gt,t) in zip(get_tensors, tensors)]
    return put_op, get_tensors

def random_crop_and_resize_image(image, bbox, height, width):
    with tf.name_scope('random_crop_and_resize'):
        if FLAGS.eval:
            image = tf.image.central_crop(image, 224./256.)
        else:
            bbox_begin, bbox_size, distorted_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=bbox,
                min_object_covered=0.1,
                aspect_ratio_range=[0.8, 1.25],
                area_range=[0.1, 1.0],
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)
            # Crop the image to the distorted bounding box
            image = tf.slice(image, bbox_begin, bbox_size)
        # Resize to the desired output size
        image = tf.image.resize_images(
            image,
            [height, width],
            tf.image.ResizeMethod.BILINEAR,
            align_corners=False)
        image.set_shape([height, width, 3])
        return image

def decode_jpeg(imgdata, channels=3):
    return tf.image.decode_jpeg(imgdata, channels=channels,
                                fancy_upscaling=False,
                                dct_method='INTEGER_FAST')

def decode_png(imgdata, channels=3):
    return tf.image.decode_png(imgdata, channels=channels)


def deserialize_image_record(record):
    feature_map = {
        'image/encoded':          tf.FixedLenFeature([ ], tf.string, ''),
        'image/class/label':      tf.FixedLenFeature([1], tf.int64,  -1),
        'image/class/text':       tf.FixedLenFeature([ ], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label   = tf.cast(obj['image/class/label'], tf.int32)
        bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                            for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
        text    = obj['image/class/text']
        return imgdata, label, bbox, text

class FeedForwardEvaluator(object):
    def __init__(self, preprocessor, eval_func):
        self.eval_func          = eval_func
        self.image_preprocessor = preprocessor
    def evaluation_step(self, total_batch_size, devices):
        preload_ops = [] # CPU pre-load
        gpucopy_ops = [] # H2D transfer
        tower_top1s = []
        with tf.device('/cpu:0'):
            dev_images, dev_labels = self.image_preprocessor.device_minibatches(
                total_batch_size)
        # Each device has its own copy of the model, referred to as a tower
        for device_num, device in enumerate(devices):
            images, labels = dev_images[device_num], dev_labels[device_num]
            with tf.device('/cpu:0'):
                # Stage images on the host
                preload_op, (images, labels) = stage([images, labels])
                preload_ops.append(preload_op)
            with tf.device(device):
                # Copy images from host to device
                gpucopy_op, (images, labels) = stage([images, labels])
                gpucopy_ops.append(gpucopy_op)
                # Evaluate the loss and compute the gradients
                with tf.variable_scope('GPU_%i' % device_num) as var_scope, \
                     tf.name_scope('tower_%i' % device_num):
                    top1 = self.eval_func(images, labels, var_scope)
                    print('>>> labels = ', labels)
                    tower_top1s.append(top1)
        # Average the topN from each tower
        with tf.device('/cpu:0'):
            total_top1 = tf.reduce_mean(tower_top1s)
        self.enqueue_ops = [tf.group(*preload_ops),
                            tf.group(*gpucopy_ops)]
        return total_top1, self.enqueue_ops
    def prefill_pipeline(self, sess):
        # Pre-fill the input pipeline with data
        for i in range(len(self.enqueue_ops)):
            sess.run(self.enqueue_ops[:i+1])


class ImagePreprocessor(object):
    def __init__(self, height, width, subset='train', dtype=tf.uint8):
        self.height = height
        self.width  = width
        self.num_devices = FLAGS.num_gpus
        self.subset = subset
        self.dtype = dtype
        self.nsummary = 10 # Max no. images to generate summaries for
    def preprocess(self, imgdata, bbox, thread_id):
        with tf.name_scope('preprocess_image'):
            try:
                image = decode_jpeg(imgdata)
            except:
                image = decode_png(imgdata)
            if thread_id < self.nsummary:
                image_with_bbox = tf.image.draw_bounding_boxes(
                    tf.expand_dims(tf.to_float(image), 0), bbox)
                tf.summary.image('original_image_and_bbox', image_with_bbox)
            image = random_crop_and_resize_image(image, bbox,
                                                 self.height, self.width)
            if thread_id < self.nsummary:
                tf.summary.image('cropped_resized_image',
                                 tf.expand_dims(image, 0))
        return image
    def device_minibatches(self, total_batch_size):
        record_input = data_flow_ops.RecordInput(
            file_pattern=os.path.join(FLAGS.data_dir, '%s-*' % self.subset),
            parallelism=64,
            # Note: This causes deadlock during init if larger than dataset
            buffer_size=FLAGS.input_buffer_size,
            batch_size=total_batch_size)
        records = record_input.get_yield_op()
        # Split batch into individual images
        records = tf.split(records, total_batch_size, 0)
        records = [tf.reshape(record, []) for record in records]
        # Deserialize and preprocess images into batches for each device
        images = defaultdict(list)
        labels = defaultdict(list)
        with tf.name_scope('input_pipeline'):
            for i, record in enumerate(records):
                imgdata, label, bbox, text = deserialize_image_record(record)
                image = self.preprocess(imgdata, bbox, thread_id=i)
                label -= 1 # Change to 0-based (don't use background class)
                device_num = i % self.num_devices
                images[device_num].append(image)
                labels[device_num].append(label)
            # Stack images back into a sub-batch for each device
            for device_num in range(self.num_devices):
                images[device_num] = tf.parallel_stack(images[device_num])
                labels[device_num] = tf.concat(labels[device_num], 0)
                images[device_num] = tf.reshape(images[device_num],
                                                [-1, self.height, self.width, 3])
                images[device_num] = tf.clip_by_value(images[device_num], 0., 255.)
                images[device_num] = tf.cast(images[device_num], self.dtype)
        return images, labels


#googlenet 
def inference_googlenet(net, input_layer):
    net.use_batch_norm = False
    def inception_v1(net, x, k, l, m, n, p, q):
        cols = [[('conv', k, (1,1))],
                [('conv', l, (1,1)), ('conv', m, (3,3))],
                [('conv', n, (1,1)), ('conv', p, (5,5))],
                [('pool', 'MAX', (3,3), (1,1), 'SAME'), ('conv', q, (1,1))]]
        return net.inception_module(x, 'incept_v1', cols)
    print('>>> input_layer=',input_layer)
    x = net.input_layer(input_layer)
    print('>>> x=',x)
    x = net.conv(x,    64, (7,7), (2,2))
    print('>>> x=',x)
    x = net.pool(x, 'MAX', (3,3), padding='SAME')
    print('>>> x=',x)
    x = net.conv(x,    64, (1,1))
    print('>>> x=',x)
    x = net.conv(x,   192, (3,3))
    print('>>> x=',x)
    x = net.pool(x, 'MAX', (3,3), padding='SAME')
    print('>>> x=',x)
    x = inception_v1(net, x,  64,  96, 128, 16,  32,  32)
    print('>>> x=',x)
    x = inception_v1(net, x, 128, 128, 192, 32,  96,  64)
    print('>>> x=',x)
    x = net.pool(x, 'MAX', (3,3), padding='SAME')
    print('>>> x=',x)
    x = inception_v1(net, x, 192,  96, 208, 16,  48,  64)
    print('>>> x=',x)
    x = inception_v1(net, x, 160, 112, 224, 24,  64,  64)
    print('>>> x=',x)
    x = inception_v1(net, x, 128, 128, 256, 24,  64,  64)
    x = inception_v1(net, x, 112, 144, 288, 32,  64,  64)
    x = inception_v1(net, x, 256, 160, 320, 32, 128, 128)
    x = net.pool(x, 'MAX', (3,3), padding='SAME')
    x = inception_v1(net, x, 256, 160, 320, 32, 128, 128)
    x = inception_v1(net, x, 384, 192, 384, 48, 128, 128)
    x = net.spatial_avg(x)
    return x

def run_evaluation(nstep, sess, top1_op, enqueue_ops):
    print("Evaluating")
    top1s = []
    print("  Step  Top-1  ")
    for step in range(nstep):
        try:
            top1 = sess.run([top1_op, enqueue_ops])[:1]
            #print('top1=',top1)
            if step == 0 or (step+1) % FLAGS.display_every == 0:
                print("% 6i %5.1f%% " % (step+1, top1[0]*100.0 ))
            top1s.append(top1)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
    nstep = len(top1s)
    if nstep == 0:
        return
    top1s = np.asarray(top1s) * 100.
    top1_mean = np.mean(top1s)
    if nstep > 2:
        top1_uncertainty = np.std(top1s, ddof=1) / np.sqrt(float(nstep))
    else:
        top1_uncertainty = float('nan')
    top1_madstd = 1.4826*np.median(np.abs(top1s - np.median(top1s)))
    print('-' * 64)
    print('Validation Top-1: %.3f %% +/- %.2f (jitter = %.1f)' % (
        top1_mean, top1_uncertainty, top1_madstd))
    print('-' * 64)

def get_num_records(tf_record_pattern):
    def count_records(tf_record_filename):
        count = 0
        for _ in tf.python_io.tf_record_iterator(tf_record_filename):
            count += 1
        return count
    filenames = sorted(tf.gfile.Glob(tf_record_pattern))
    nfile = len(filenames)
    return (count_records(filenames[0])*(nfile-1) +
            count_records(filenames[-1]))

def main():
    tf.set_random_seed(1234)
    np.random.seed(4321)
    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Basic options

    cmdline.add_argument('--data_dir', default=None,
                         help="""Path to dataset in TFRecord format
                         (aka Example protobufs). Files should be
                         named 'train-*' and 'validation-*'.""")
    cmdline.add_argument('-b', '--batch_size', default=64, type=int,
                         help="""Size of each minibatch.""")
    cmdline.add_argument('--num_batches', default=50000, type=int,
                         help="""Number of batches to run.""")
    cmdline.add_argument('--num_epochs', default=100, type=int,
                         help="""Number of epochs to run
                         (overrides --num_batches).""")
    cmdline.add_argument('-g', '--num_gpus', default=1, type=int,
                         help="""Number of GPUs to run on.""")
    cmdline.add_argument('--log_dir', default="./log_dir",
                         help="""Directory in which to write training
                         summaries and checkpoints.""")
    cmdline.add_argument('--display_every', default=10, type=int,
                         help="""How often (in iterations) to print out
                         running information.""")

    global FLAGS
    FLAGS, unknown_args = cmdline.parse_known_args()
    FLAGS.strong_scaling = False
    FLAGS.nccl           = True
    FLAGS.xla            = False
    FLAGS.num_gpus = 1
    FLAGS.eval = True
    print('FLAGS.eval=', FLAGS.eval)

    nclass = 2
    total_batch_size = FLAGS.batch_size
    if not FLAGS.strong_scaling:
        total_batch_size *= FLAGS.num_gpus
    devices = ['/gpu:%i' % i for i in range(FLAGS.num_gpus)]
    subset = 'validation' 
    print('subset=', subset)

    print("Cmd line args:")
    print('\n'.join(['  '+arg for arg in sys.argv[1:]]))

    if FLAGS.data_dir is not None and FLAGS.data_dir != '':
        nrecord = get_num_records(os.path.join(FLAGS.data_dir, '%s-*' % subset))
    else:
        nrecord = FLAGS.num_batches * total_batch_size
    FLAGS.input_buffer_size     = min(10000, nrecord)

    print('>> total_batch_size=', total_batch_size)
    print('>> nrecord=',nrecord)
    model_dtype = tf.float32

    if FLAGS.num_epochs is not None:
        if FLAGS.data_dir is None:
            raise ValueError("num_epochs requires data_dir to be specified")
        nstep = nrecord * FLAGS.num_epochs // total_batch_size
    else:
        nstep = FLAGS.num_batches
        FLAGS.num_epochs = max(nstep * total_batch_size // nrecord, 1)

    height, width = 224, 224
    model_func = inference_googlenet
    FLAGS.learning_rate = 0.04

    if FLAGS.data_dir is None:
        preprocessor = DummyPreprocessor(height, width, total_batch_size//len(devices), nclass)
    else:
        preprocessor = ImagePreprocessor(height, width, subset)

    def eval_func(images, labels, var_scope):
        net = GPUNetworkBuilder(
            False, dtype=model_dtype, use_xla=FLAGS.xla)
        output = model_func(net, images)
        logits = net.fully_connected(output, nclass, activation='LINEAR')
        if logits.dtype != tf.float32:
            logits = tf.cast(logits, tf.float32)
        with tf.device('/cpu:0'):
            top1 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
        return top1
    def eval_func1(images, labels, var_scope):
        net = GPUNetworkBuilder(
            False, dtype=model_dtype, use_xla=FLAGS.xla)
        output = model_func(net, images)
        logits = net.fully_connected(output, nclass, activation='LINEAR')
        if logits.dtype != tf.float32:
            logits = tf.cast(logits, tf.float32)
        with tf.device('/cpu:0'):
            top1 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
        return top1,logits 

    # return logits given images
    def eval_f(images,var_scope):
        net = GPUNetworkBuilder(
            False, dtype=model_dtype, use_xla=FLAGS.xla)
        output = model_func(net, images)
        logits = net.fully_connected(output, 2, activation='LINEAR')
        if logits.dtype != tf.float32:
            logits = tf.cast(logits, tf.float32)
        return logits 

    if FLAGS.data_dir is None:
        raise ValueError("eval requires data_dir to be specified")
    evaluator = FeedForwardEvaluator(preprocessor, eval_func)
    print("Building evaluation graph")
    top1_op, enqueue_ops = evaluator.evaluation_step(
            total_batch_size, devices)

    print("Creating session")
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True))
    config.intra_op_parallelism_threads = 1

    sess = tf.Session(config=config)

    train_writer = None
    saver = None
    summary_ops = None

    print('saver=',saver)
    if len(FLAGS.log_dir):
        log_dir = FLAGS.log_dir
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        summary_ops = tf.summary.merge_all()
        last_summary_time = time.time()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=3)
        last_save_time = time.time()

    restored = False
    if saver is not None:
        print('check point state - recover')
        ckpt = tf.train.get_checkpoint_state(log_dir)
        checkpoint_file = os.path.join(log_dir, "checkpoint")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            restored = True
            print("Restored session from checkpoint " + ckpt.model_checkpoint_path)
        else:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

    if FLAGS.eval:
        if not restored:
            raise ValueError("No checkpoint found for evaluation")
        else:
            print("Pre-filling input pipeline")
#            evaluator.prefill_pipeline(sess)
            nstep = nrecord // total_batch_size
            print('eval nstep = ', nstep)
            nstep = nstep*10
            print('doubled eval nstep = ', nstep)

#            filenames = tf.train.match_filenames_once('/home/sun/work/nvcnn-ex/female_20/*.jpg')
#            print('filenames=',filenames)
#            file = open('./list','r')
#            list = file.read()
#            print(list)
#            filename_queue = tf.train.string_input_producer( tf.string_split(list,'\n') )
            filename_queue = tf.train.string_input_producer( ["./female_20/female_20_00373.jpg", 'female_20/female_20_00300.jpg', 'female_20/female_20_00301.jpg', 'female_20/female_20_00302.jpg', 'female_20/female_20_00303.jpg', 'female_20/female_20_00304.jpg', 'female_20/female_20_00305.jpg', 'female_20/female_20_00306.jpg', 'female_20/female_20_00307.jpg', 'female_20/female_20_00308.jpg', 'female_20/female_20_00309.jpg', 'female_20/male_20_00360.jpg', 'female_20/male_20_00361.jpg', 'female_20/male_20_00362.jpg', 'female_20/male_20_00363.jpg', 'female_20/male_20_00364.jpg', 'female_20/male_20_00365.jpg', 'female_20/male_20_00366.jpg', 'female_20/male_20_00367.jpg', 'female_20/male_20_00368.jpg', 'female_20/male_20_00369.jpg'] )
            img_reader = tf.WholeFileReader()
            key,image_file = img_reader.read(filename_queue)
            image_d = tf.image.decode_jpeg(image_file, channels=3)
            print('>> image=',image_d)
#            image_f = tf.image.convert_image_dtype(image_d,dtype=tf.float32)
#            print('>> image=',image_f)
            image_c = tf.image.central_crop(image_d, 224./256.)
            print('>> image=',image_c)
            image_r = tf.image.resize_images(image_c, [224, 224], tf.image.ResizeMethod.BILINEAR, align_corners=False)
            print('>> image=',image_r)
            image_r.set_shape([224, 224, 3])
            print('>> image=',image_r)

            image_rs = tf.reshape(image_r,[1,224,224,3])
            print('>> image=',image_rs)
            device_num=0
            with tf.variable_scope('GPU_%i' % device_num,reuse=True) as var_scope, \
                     tf.name_scope('tower_%i' % device_num):
#                top1,logits2 = self.eval_func(image, tf.convert_to_tensor([[1.0]],dtype=float32), var_scope)
#                top1 = eval_func(image, 1, var_scope)
#                top1,logits2 = eval_func1(image, 1, var_scope)
	        logits = eval_f(image_rs,var_scope)

            print('logits=',logits)

            with sess:
            # Coordinate the loading of image files.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

            # Get an image tensor and print its value.
            #image_tensor, logits_tensor = sess.run([image, logits])
                for i in range(20):
                   logits_tensor = sess.run([logits])
                   #print('logits=',logits_tensor)
                   # image recover float32 -> uint8
                   img = image_d.eval()
                   img_f = image_c.eval()
                   img_f = (img_f*255).round().astype(np.uint8)
                   print('i=',i,key.eval(), 'logits=',softmax(logits_tensor[0]))
                   print(img.shape)
                   cv2.imwrite('./female_20/' + str(i) + '.jpg',img)
                   cv2.imwrite('./female_20/' + str(i) + '_f.jpg',img_f)
 
#                   top1_tensor, logits2_tensor = sess.run([top1,logits2])
#                   print(key.eval(), 'top1=',top1_tensor,'logits2=',logits2_tensor)
            # Finish off the filename queue coordinator.
                coord.request_stop()
                coord.join(threads)

            #run_evaluation(nstep, sess, top1_op, enqueue_ops)
            return

def softmax(w, t = 1.0):
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist

if __name__ == '__main__':
    npa = np.array
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()

