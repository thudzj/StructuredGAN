import os, sys, time
import numpy as np

import scipy
from collections import OrderedDict
import pickle

from lasagne.layers import InputLayer, ReshapeLayer, FlattenLayer, Upscale2DLayer, MaxPool2DLayer, DropoutLayer, ConcatLayer, DenseLayer, NINLayer
from lasagne.layers import GaussianNoiseLayer, Conv2DLayer, Pool2DLayer, GlobalPoolLayer, NonlinearityLayer, FeaturePoolLayer, DimshuffleLayer, ElemwiseSumLayer
from lasagne.utils import floatX
from zca_bn import ZCA
from zca_bn import mean_only_bn as WN

import gzip, os, cPickle, time, math, argparse, shutil, sys

import numpy as np
import theano, lasagne
import theano.tensor as T
import lasagne.layers as ll
import lasagne.nonlinearities as ln
from lasagne.layers import dnn
import nn
from lasagne.init import Normal
from theano.sandbox.rng_mrg import MRG_RandomStreams
import cifar10_data

from layers.merge import ConvConcatLayer, MLPConcatLayer
from layers.deconv import Deconv2DLayer

from components.shortcuts import convlayer, mlplayer
from components.objectives import categorical_crossentropy_ssl_separated, maximum_mean_discripancy, categorical_crossentropy, feature_matching, mean_squared_error
from utils.create_ssl_data import create_ssl_data, create_ssl_data_subset
from utils.others import get_nonlin_list, get_pad_list, bernoullisample, printarray_2D, array2file_2D
import utils.paramgraphics as paramgraphics

def build_network():
    conv_defs = {
        'W': lasagne.init.HeNormal('relu'),
        'b': lasagne.init.Constant(0.0),
        'filter_size': (3, 3),
        'stride': (1, 1),
        'nonlinearity': lasagne.nonlinearities.LeakyRectify(0.1)
    }

    nin_defs = {
        'W': lasagne.init.HeNormal('relu'),
        'b': lasagne.init.Constant(0.0),
        'nonlinearity': lasagne.nonlinearities.LeakyRectify(0.1)
    }

    dense_defs = {
        'W': lasagne.init.HeNormal(1.0),
        'b': lasagne.init.Constant(0.0),
        'nonlinearity': lasagne.nonlinearities.softmax
    }

    wn_defs = {
        'momentum': .999
    }

    net = InputLayer        (     name='input',    shape=(None, 3, 32, 32))
    net = GaussianNoiseLayer(net, name='noise',    sigma=.15)
    net = WN(Conv2DLayer    (net, name='conv1a',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    net = WN(Conv2DLayer    (net, name='conv1b',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    net = WN(Conv2DLayer    (net, name='conv1c',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    net = MaxPool2DLayer    (net, name='pool1',    pool_size=(2, 2))
    net = DropoutLayer      (net, name='drop1',    p=.5)
    net = WN(Conv2DLayer    (net, name='conv2a',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    net = WN(Conv2DLayer    (net, name='conv2b',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    net = WN(Conv2DLayer    (net, name='conv2c',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    net = MaxPool2DLayer    (net, name='pool2',    pool_size=(2, 2))
    net = DropoutLayer      (net, name='drop2',    p=.5)
    net = WN(Conv2DLayer    (net, name='conv3a',   num_filters=512, pad=0,      **conv_defs), **wn_defs)
    net = WN(NINLayer       (net, name='conv3b',   num_units=256,               **nin_defs),  **wn_defs)
    net = WN(NINLayer       (net, name='conv3c',   num_units=128,               **nin_defs),  **wn_defs)
    net = GlobalPoolLayer   (net, name='pool3')
    net = WN(DenseLayer     (net, name='dense',    num_units=10,       **dense_defs), **wn_defs)

    return net

def rampup(epoch):
    if epoch < 80:
        p = max(0.0, float(epoch)) / float(80)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown(epoch):
    if epoch >= (300 - 50):
        ep = (epoch - (300 - 50)) * 0.5
        return math.exp(-(ep * ep) / 50)
    else:
        return 1.0

def robust_adam(loss, params, learning_rate, beta1=0.9, beta2=0.999, epsilon=1.0e-8):
    # Convert NaNs to zeros.
    def clear_nan(x):
        return T.switch(T.isnan(x), np.float32(0.0), x)

    new = OrderedDict()
    pg = zip(params, lasagne.updates.get_or_compute_grads(loss, params))
    t = theano.shared(lasagne.utils.floatX(0.))

    new[t] = t + 1.0
    coef = learning_rate * T.sqrt(1.0 - beta2**new[t]) / (1.0 - beta1**new[t])
    for p, g in pg:
        value = p.get_value(borrow=True)
        m = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
        v = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
        new[m] = clear_nan(beta1 * m + (1.0 - beta1) * g)
        new[v] = clear_nan(beta2 * v + (1.0 - beta2) * g**2)
        new[p] = clear_nan(p - coef * new[m] / (T.sqrt(new[v]) + epsilon))

    return new

'''
parameters
'''
# global
parser = argparse.ArgumentParser()
parser.add_argument("-key", type=str, default=argparse.SUPPRESS)
parser.add_argument("-ssl_seed", type=int, default=1)
parser.add_argument("-nlabeled", type=int, default=4000)
parser.add_argument("-oldmodel", type=str, default=argparse.SUPPRESS)
args = parser.parse_args()
args = vars(args).items()
cfg = {}
for name, val in args:
    cfg[name] = val

filename_script=os.path.basename(os.path.realpath(__file__))
outfolder=os.path.join("results-ssl", os.path.splitext(filename_script)[0])
outfolder+='.'
for item in cfg:
    if item is not 'oldmodel':
        outfolder += item+str(cfg[item])+'.'
    else:
        outfolder += 'oldmodel.'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
sample_path = os.path.join(outfolder, 'sample')
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
logfile=os.path.join(outfolder, 'logfile.log')
shutil.copy(os.path.realpath(__file__), os.path.join(outfolder, filename_script))
# fixed random seeds
ssl_data_seed=cfg['ssl_seed']
num_labelled=cfg['nlabeled']
print ssl_data_seed, num_labelled

seed=1234
rng=np.random.RandomState(seed)
theano_rng=MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# C
scaled_unsup_weight_max = 100.0
# G
n_z=100
# optimization
b1=.5 # mom1 in Adam
batch_size=200
batch_size_eval=200
lr=3e-4
cla_lr=3e-3
num_epochs=1000
pre_num_epoch=20
anneal_lr_epoch=200
anneal_lr_every_epoch=1
anneal_lr_factor_cla=.99
anneal_lr_factor=.995
# data dependent
gen_final_non=ln.tanh
num_classes=10
dim_input=(32,32)
in_channels=3
colorImg=True
generation_scale=True
z_generated=num_classes
# evaluation
vis_epoch=1
eval_epoch=1

'''
data
'''
def rescale(mat):
    return np.cast[theano.config.floatX](mat)

train_x, train_y = cifar10_data.load('./cifar10/','train')
eval_x, eval_y = cifar10_data.load('./cifar10/','test')

train_y = np.int32(train_y)
eval_y = np.int32(eval_y)
train_x = rescale(train_x)
eval_x = rescale(eval_x)
x_unlabelled = train_x.copy()

rng_data = np.random.RandomState(ssl_data_seed)
inds = rng_data.permutation(train_x.shape[0])
train_x = train_x[inds]
train_y = train_y[inds]
x_labelled = []
y_labelled = []
for j in range(num_classes):
    x_labelled.append(train_x[train_y==j][:num_labelled/num_classes])
    y_labelled.append(train_y[train_y==j][:num_labelled/num_classes])
x_labelled = np.concatenate(x_labelled, axis=0)
y_labelled = np.concatenate(y_labelled, axis=0)
del train_x

if True:
    print 'Size of training data', x_labelled.shape[0], x_unlabelled.shape[0]
    y_order = np.argsort(y_labelled)
    _x_mean = x_labelled[y_order]
    image = paramgraphics.mat_to_img(_x_mean.T, dim_input, tile_shape=(num_classes, num_labelled/num_classes), colorImg=colorImg, scale=generation_scale, save_path=os.path.join(outfolder, 'x_l_'+str(ssl_data_seed)+'_sgan.png'))

num_batches_l = x_labelled.shape[0] / batch_size
num_batches_u = x_unlabelled.shape[0] / batch_size
num_batches_e = eval_x.shape[0] / batch_size_eval


'''
models
'''
# symbols
sym_z_image = T.tile(theano_rng.uniform((z_generated, n_z)), (num_classes, 1))
sym_z_rand = theano_rng.uniform(size=(batch_size, n_z))
sym_y_g = T.ivector()

sym_x_l = T.tensor4()
sym_y = T.ivector()

sym_x_u_d = T.tensor4()

sym_z_m = T.matrix()
sym_y_m = T.ivector()

sym_x_u = T.tensor4()
sym_x_u_rep = T.tensor4('two_pass')
sym_x_u_i = T.tensor4()

sym_x_eval = T.tensor4()
sym_lr = T.scalar()

# te
sym_lr_cla = T.scalar('separate_lr')
sym_unsup_weight = T.scalar('unsup_weight')
sym_b_c = T.scalar('adam_beta1')
sym_w_g = T.scalar('w_g')

shared_unlabel = theano.shared(x_unlabelled, borrow=True)
slice_x_u_d = T.ivector()
slice_x_u_c = T.ivector()
slice_x_u_i = T.ivector()

classifier = build_network()

# generator y2x: p_g(x, y) = p(y) p_g(x | y) where x = G(z, y), z follows p_g(z)
gen_in_z = ll.InputLayer(shape=(None, n_z))
gen_in_y = ll.InputLayer(shape=(None,))
gen_layers = [gen_in_z]
gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-00'))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu, name='gen-01'), g=None, name='gen-02'))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (-1,512,4,4), name='gen-03'))
gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-10'))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='gen-11'), g=None, name='gen-12')) # 4 -> 8
gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-20'))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='gen-21'), g=None, name='gen-22')) # 8 -> 16
gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-30'))
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (None,3,32,32), (5,5), W=Normal(0.05), nonlinearity=gen_final_non, name='gen-31'), train_g=True, init_stdv=0.1, name='gen-32')) # 16 -> 32

# discriminator xy2p: test a pair of input comes from p(x, y) instead of p_c or p_g
dis_in_x = ll.InputLayer(shape=(None, in_channels) + dim_input)
dis_in_y = ll.InputLayer(shape=(None,))
dis_layers = [dis_in_x]
dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-00'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-01'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 32, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-02'), name='dis-03'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-20'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 32, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-21'), name='dis-22'))
dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-23'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-30'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 64, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-31'), name='dis-32'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-40'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 64, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-41'), name='dis-42'))
dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-43'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-50'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 128, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-51'), name='dis-52'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-60'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 128, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-61'), name='dis-62'))
dis_layers.append(ll.GlobalPoolLayer(dis_layers[-1], name='dis-63'))
dis_layers.append(MLPConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-70'))
dis_layers.append(nn.weight_norm(ll.DenseLayer(dis_layers[-1], num_units=1, W=Normal(0.05), nonlinearity=ln.sigmoid, name='dis-71'), train_g=True, init_stdv=0.1, name='dis-72'))

# inference module
inf_in_x = ll.InputLayer(shape=(None, in_channels) + dim_input)
inf_layers = [inf_in_x]
inf_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(inf_layers[-1], 64, (4,4), stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='inf-02'), name='inf-03'))
inf_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(inf_layers[-1], 128, (4,4), stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='inf-11'), name='inf-12'))
inf_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(inf_layers[-1], 256, (4,4), stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='inf-21'), name='inf-22'))
inf_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(inf_layers[-1], 512, (4,4), stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='inf-31'), name='inf-32'))
inf_layers.append(ll.DenseLayer(inf_layers[-1], num_units=n_z, W=Normal(0.05), nonlinearity=None, name='inf-4'))

# discriminator xz
disxz_in_x = ll.InputLayer(shape=(None, in_channels) + dim_input)
disxz_in_z = ll.InputLayer(shape=(None, n_z))
disxz_z_layers = [disxz_in_z]
disxz_z_layers.append(ll.DenseLayer(disxz_z_layers[-1], num_units=512, W=Normal(0.05), nonlinearity=nn.lrelu, name='disxz-0'))
disxz_z_layers.append(ll.DenseLayer(disxz_z_layers[-1], num_units=512, W=Normal(0.05), nonlinearity=nn.lrelu, name='disxz-1'))
disxz_x_layers = [disxz_in_x]
disxz_x_layers.append(dnn.Conv2DDNNLayer(disxz_x_layers[-1], 128, (5,5), stride=2, pad='same', W=Normal(0.05), nonlinearity=nn.lrelu, name='disxz-2'))
disxz_x_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(disxz_x_layers[-1], 256, (5,5), stride=2, pad='same', W=Normal(0.05), nonlinearity=nn.lrelu, name='disxz-31'), name='disxz-32'))
disxz_x_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(disxz_x_layers[-1], 512, (5,5), stride=2, pad='same', W=Normal(0.05), nonlinearity=nn.lrelu, name='disxz-41'), name='disxz-42'))
disxz_layers = [ll.ConcatLayer([ll.FlattenLayer(disxz_x_layers[-1]), disxz_z_layers[-1]], name='disxz-5')]
disxz_layers.append(ll.DenseLayer(disxz_layers[-1], num_units=1024, W=Normal(0.05), nonlinearity=nn.lrelu, name='disxz-6'))
disxz_layers.append(ll.DenseLayer(disxz_layers[-1], num_units=1, W=Normal(0.05), nonlinearity=ln.sigmoid, name='disxz-7'))

'''
objectives
'''
# zca
whitener = ZCA(x=x_unlabelled)
sym_x_l_zca = whitener.apply(sym_x_l)
sym_x_eval_zca = whitener.apply(sym_x_eval)
sym_x_u_zca = whitener.apply(sym_x_u)
sym_x_u_rep_zca = whitener.apply(sym_x_u_rep)
sym_x_u_d_zca = whitener.apply(sym_x_u_d)

# init
lasagne.layers.get_output(classifier, sym_x_u_zca, init=True)
init_updates = [u for l in lasagne.layers.get_all_layers(classifier) for u in getattr(l, 'init_updates', [])]
init_fn = theano.function([sym_x_u], [], updates=init_updates)

# outputs
gen_out_x = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_rand}, deterministic=False)
gen_out_x_zca = whitener.apply(gen_out_x)
image = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_image}, deterministic=False) # for generation
gen_out_x_m = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_m, gen_in_z:sym_z_m}, deterministic=False)
gen_out_x_m_zca = whitener.apply(gen_out_x_m)

cla_out_y_l = ll.get_output(classifier, sym_x_l_zca, deterministic=False)
cla_out_y_eval = ll.get_output(classifier, sym_x_eval_zca, deterministic=True)
cla_out_y = ll.get_output(classifier, sym_x_u_zca, deterministic=False)
cla_out_y_rep = ll.get_output(classifier, sym_x_u_rep_zca, deterministic=False)
bn_updates = [u for l in lasagne.layers.get_all_layers(classifier) for u in getattr(l, 'bn_updates', [])]
cla_out_y_m = ll.get_output(classifier, gen_out_x_m_zca, deterministic=False)
cla_out_y_d = ll.get_output(classifier, sym_x_u_d_zca, deterministic=False)
cla_out_y_d_hard = cla_out_y_d.argmax(axis=1)
cla_out_y_g = ll.get_output(classifier, gen_out_x_zca, deterministic=False)

inf_z = ll.get_output(inf_layers[-1], {inf_in_x:sym_x_u_i}, deterministic=False)
inf_z_g = ll.get_output(inf_layers[-1], {inf_in_x:gen_out_x}, deterministic=False)

dis_out_p = ll.get_output(dis_layers[-1], {dis_in_x:T.concatenate([sym_x_l, sym_x_u_d, gen_out_x_m], axis=0)[:batch_size],dis_in_y:T.concatenate([sym_y, cla_out_y_d_hard, sym_y_m], axis=0)[:batch_size]}, deterministic=False)
dis_out_p_g = ll.get_output(dis_layers[-1], {dis_in_x:gen_out_x,dis_in_y:sym_y_g}, deterministic=False)

disxz_out_p = ll.get_output(disxz_layers[-1], {disxz_in_x:sym_x_u_i, disxz_in_z: inf_z}, deterministic=False)
disxz_out_p_g = ll.get_output(disxz_layers[-1], {disxz_in_x:gen_out_x, disxz_in_z: sym_z_rand}, deterministic=False)

accurracy_eval = (lasagne.objectives.categorical_accuracy(cla_out_y_eval, sym_y)) # for evaluation
accurracy_eval = accurracy_eval.mean()

# costs
bce = lasagne.objectives.binary_crossentropy

dis_cost_p = bce(dis_out_p, T.ones(dis_out_p.shape)).mean() # D distincts p
dis_cost_p_g = bce(dis_out_p_g, T.zeros(dis_out_p_g.shape)).mean() # D distincts p_g
gen_cost_p_g_1 = bce(dis_out_p_g, T.ones(dis_out_p_g.shape)).mean() # G fools D

disxz_cost_p = bce(disxz_out_p, T.ones(disxz_out_p.shape)).mean()
disxz_cost_p_g = bce(disxz_out_p_g, T.zeros(disxz_out_p_g.shape)).mean()
inf_cost_p_i = bce(disxz_out_p, T.zeros(disxz_out_p.shape)).mean()
gen_cost_p_g_2 = bce(disxz_out_p_g, T.ones(disxz_out_p_g.shape)).mean()

cla_cost_l = T.mean(lasagne.objectives.categorical_crossentropy(cla_out_y_l, sym_y), dtype=theano.config.floatX, acc_dtype=theano.config.floatX)
cla_cost_g = T.mean(lasagne.objectives.categorical_crossentropy(cla_out_y_m, sym_y_m), dtype=theano.config.floatX, acc_dtype=theano.config.floatX) * sym_w_g
cla_cost_u = sym_unsup_weight * T.mean(lasagne.objectives.squared_error(cla_out_y, cla_out_y_rep), dtype=theano.config.floatX, acc_dtype=theano.config.floatX)

rz = mean_squared_error(inf_z_g, sym_z_rand, n_z)
ry = categorical_crossentropy(cla_out_y_g, sym_y_g)

pretrain_cost = cla_cost_l + cla_cost_u
cla_cost = cla_cost_l + cla_cost_u + cla_cost_g
dis_cost = dis_cost_p + dis_cost_p_g
disxz_cost = disxz_cost_p + disxz_cost_p_g
inf_cost = inf_cost_p_i + rz
gen_cost = gen_cost_p_g_1 + gen_cost_p_g_2 + rz + ry

dis_cost_list=[dis_cost + disxz_cost, dis_cost, dis_cost_p, dis_cost_p_g, disxz_cost, disxz_cost_p, disxz_cost_p_g]
gen_cost_list=[gen_cost, gen_cost_p_g_1, gen_cost_p_g_2, rz, ry]
inf_cost_list=[inf_cost, inf_cost_p_i, rz]
cla_cost_list=[cla_cost, cla_cost_l, cla_cost_u, cla_cost_g]

# updates of D
dis_params = ll.get_all_params(dis_layers, trainable=True) + ll.get_all_params(disxz_layers, trainable=True)
dis_grads = T.grad(dis_cost+disxz_cost, dis_params)
dis_updates = lasagne.updates.adam(dis_grads, dis_params, beta1=b1, learning_rate=sym_lr)

# updates of G
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_grads = T.grad(gen_cost, gen_params)
gen_updates = lasagne.updates.adam(gen_grads, gen_params, beta1=b1, learning_rate=sym_lr)

# updates of C
cla_params = ll.get_all_params(classifier, trainable=True)
cla_updates_ = robust_adam(cla_cost, cla_params, learning_rate=sym_lr_cla, beta1=sym_b_c, beta2=.999, epsilon=1e-8)
pretrain_updates_ = robust_adam(pretrain_cost, cla_params, learning_rate=sym_lr_cla, beta1=sym_b_c, beta2=.999, epsilon=1e-8) 

# I
inf_params = ll.get_all_params(inf_layers, trainable=True)
inf_grads = T.grad(inf_cost, inf_params)
inf_updates = lasagne.updates.adam(inf_grads, inf_params, beta1=b1, learning_rate=sym_lr)

######## avg
avg_params = lasagne.layers.get_all_params(classifier)
cla_param_avg=[]
for param in avg_params:
   value = param.get_value(borrow=True)
   cla_param_avg.append(theano.shared(np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable,
                              name=param.name))
cla_avg_updates = [(a,a + 0.01*(p-a)) for p,a in zip(avg_params,cla_param_avg)]
cla_avg_givens = [(p,a) for p,a in zip(avg_params, cla_param_avg)]
cla_updates = cla_updates_.items() + bn_updates + cla_avg_updates
pretrain_updates = pretrain_updates_.items() + bn_updates + cla_avg_updates

# functions
train_batch_dis = theano.function(inputs=[sym_x_l, sym_y, slice_x_u_d, sym_y_m, sym_z_m, slice_x_u_i, sym_y_g, sym_lr],
                                  outputs=dis_cost_list, updates=dis_updates,
                                  givens={sym_x_u_d: shared_unlabel[slice_x_u_d], sym_x_u_i: shared_unlabel[slice_x_u_i]})
train_batch_gen = theano.function(inputs=[sym_y_g, sym_lr],
                                  outputs=gen_cost_list, updates=gen_updates)
train_batch_inf = theano.function(inputs=[slice_x_u_i, sym_y_g, sym_lr],
                                  outputs=inf_cost_list, updates=inf_updates,
                                  givens={sym_x_u_i: shared_unlabel[slice_x_u_i]})
train_batch_cla = theano.function(inputs=[sym_x_l, sym_y, slice_x_u_c, sym_y_m, sym_z_m, sym_lr_cla, sym_b_c, sym_unsup_weight,sym_w_g],
                                  outputs=cla_cost_list , updates=cla_updates,
                                  givens={sym_x_u: shared_unlabel[slice_x_u_c], sym_x_u_rep: shared_unlabel[slice_x_u_c]})
pretrain_batch_cla = theano.function(inputs=[sym_x_l, sym_y, slice_x_u_c, sym_lr_cla, sym_b_c, sym_unsup_weight],
                                  outputs=[pretrain_cost, cla_cost_l, cla_cost_u], updates=pretrain_updates,
                                  givens={sym_x_u: shared_unlabel[slice_x_u_c], sym_x_u_rep: shared_unlabel[slice_x_u_c]})

generate = theano.function(inputs=[sym_y_g], outputs=image)
inference = theano.function(inputs=[sym_x_u_i], outputs=inf_z)
# avg
evaluate = theano.function(inputs=[sym_x_eval, sym_y], outputs=[accurracy_eval], givens=cla_avg_givens)

'''
Load pretrained model
'''
if 'oldmodel' in cfg:
    from utils.checkpoints import load_weights
    load_weights(cfg['oldmodel'], dis_layers+[classifier,]+gen_layers)
    for (p, a) in zip(ll.get_all_params(classifier), avg_params):
        a.set_value(p.get_value())

'''
train and evaluate
'''
init_fn(x_unlabelled[:batch_size])

'''
Pretrain C
'''
print 'Start pretraining'
for epoch in range(1, 1+pre_num_epoch):
    # randomly permute data and labels
    p_l = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[p_l]
    y_labelled = y_labelled[p_l]
    p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')

    for i in range(num_batches_u):
        i_c = i % num_batches_l
        bp = pretrain_batch_cla(x_labelled[i_c*batch_size:(i_c+1)*batch_size], y_labelled[i_c*batch_size:(i_c+1)*batch_size], p_u[i*batch_size:(i+1)*batch_size], 3e-3, 0.9, 100)
        #print(bp)
    accurracy=[]
    for i in range(num_batches_e):
        accurracy_batch = evaluate(eval_x[i*batch_size_eval:(i+1)*batch_size_eval], eval_y[i*batch_size_eval:(i+1)*batch_size_eval])
        accurracy += accurracy_batch
    accurracy=np.mean(accurracy)
    print str(epoch) + ':Pretrain accuracy: ' + str(1-accurracy)


print 'Start training'
for epoch in range(1, 1+num_epochs):
    start = time.time()

    # randomly permute data and labels
    p_l = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[p_l]
    y_labelled = y_labelled[p_l]
    p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_d = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_i = rng.permutation(x_unlabelled.shape[0]).astype('int32')

    if epoch < 500:
        if epoch % 50 == 1: # 4, 8, 12, 16
            batch_l = 200 - (epoch // 50 + 1) * 16
            batch_c = (epoch // 50 + 1) * 16
            batch_g = 1#(epoch // 50 + 1) * 10
    elif epoch < 1000 and epoch % 100 == 0:
        batch_l = 50
        batch_c = 140 - 10 * (epoch-500)/100
        batch_g = 10 + 10 * (epoch-500)/100

    dl = [0.] * len(dis_cost_list)
    gl = [0.] * len(gen_cost_list)
    cl = [0.] * len(cla_cost_list)
    il = [0.] * len(inf_cost_list)

    if (epoch % eval_epoch == 0):
        # te
        rampup_value = rampup(epoch-1)
        rampdown_value = rampdown(epoch-1)
        lr_c = cla_lr
        b1_c = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
        unsup_weight = rampup_value * scaled_unsup_weight_max if epoch > 1 else 0.
        w_g = np.float32(min(float(epoch) / 300.0, 1.0))

        size_l = 100
        size_g = 100
        size_u = 100
        for i in range(num_batches_u * eval_epoch):
            i_l = i % (x_labelled.shape[0] // size_l)
            i_u = i % (x_unlabelled.shape[0] // size_u)

            y_real = np.int32(np.random.randint(10, size=size_g))
            z_real = np.random.uniform(size=(size_g, n_z)).astype(np.float32)

            cl_b = train_batch_cla(x_labelled[i_l*size_l:(i_l+1)*size_l], y_labelled[i_l*size_l:(i_l+1)*size_l], p_u[i_u*size_u:(i_u+1)*size_u], y_real, z_real, lr_c, b1_c, unsup_weight, w_g)

            for j in xrange(len(cl)):
                cl[j] += cl_b[j]

            if i_l == ((x_labelled.shape[0] // size_l) - 1):
                p_l = rng.permutation(x_labelled.shape[0])
                x_labelled = x_labelled[p_l]
                y_labelled = y_labelled[p_l]
            if i_u == (num_batches_u - 1):
                p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')

        for i in xrange(len(cl)):
            cl[i] /= num_batches_u * eval_epoch

        accurracy=[]
        for i in range(num_batches_e):
            accurracy_batch = evaluate(eval_x[i*batch_size_eval:(i+1)*batch_size_eval], eval_y[i*batch_size_eval:(i+1)*batch_size_eval])
            accurracy += accurracy_batch
        accurracy=np.mean(accurracy)
        print ('ErrorEval=%.5f\n' % (1-accurracy,))
        with open(logfile,'a') as f:
            f.write(('ErrorEval=%.5f\n\n' % (1-accurracy,)))

    for i in range(num_batches_u):
        i_l = i % (x_labelled.shape[0] // batch_l)

        from_u_i = i*batch_size
        to_u_i = (i+1)*batch_size
        from_u_d = i*batch_c
        to_u_d = (i+1) * batch_c
        from_l = i_l*batch_l
        to_l = (i_l+1)*batch_l

        sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size/num_classes))
        y_real = np.int32(np.random.randint(10, size=batch_g))
        z_real = np.random.uniform(size=(batch_g, n_z)).astype(np.float32)

        tmp = time.time()
        dl_b = train_batch_dis(x_labelled[from_l:to_l], y_labelled[from_l:to_l], p_u_d[from_u_d:to_u_d], y_real, z_real, p_u_i[from_u_i:to_u_i], sample_y, lr)
        for j in xrange(len(dl)):
            dl[j] += dl_b[j]

        il_b = train_batch_inf(p_u_i[from_u_i:to_u_i], sample_y, lr)
        for j in xrange(len(il)):
            il[j] += il_b[j]

        gl_b = train_batch_gen(sample_y, lr)
        for j in xrange(len(gl)):
            gl[j] += gl_b[j]

        if i_l == ((x_labelled.shape[0] // batch_l) - 1):
            p_l = rng.permutation(x_labelled.shape[0])
            x_labelled = x_labelled[p_l]
            y_labelled = y_labelled[p_l]

    for i in xrange(len(dl)):
        dl[i] /= num_batches_u
    for i in xrange(len(gl)):
        gl[i] /= num_batches_u
    for i in xrange(len(cl)):
        cl[i] /= num_batches_u

    if (epoch >= anneal_lr_epoch) and (epoch % anneal_lr_every_epoch == 0):
        lr = lr*anneal_lr_factor
        cla_lr *= anneal_lr_factor_cla

    t = time.time() - start

    line = "*Epoch=%d Time=%.2f LR=%.5f\n" %(epoch, t, lr) + "DisLosses: " + str(dl)+"\nGenLosses: "+str(gl)+"\nInfLosses: "+str(il)+"\nClaLosses: "+str(cl)
    print line
    with open(logfile,'a') as f:
        f.write(line + "\n")

    # random generation for visualization
    if epoch % vis_epoch == 0:
        import  utils.paramgraphics as paramgraphics
        tail = '-'+str(epoch)+'.png'
        ran_y = np.int32(np.repeat(np.arange(num_classes), num_classes))
        x_gen = generate(ran_y)
        x_gen = x_gen.reshape((z_generated*num_classes,-1))
        image = paramgraphics.mat_to_img(x_gen.T, dim_input, colorImg=colorImg, scale=generation_scale, save_path=os.path.join(sample_path, 'sample'+tail))

    if epoch % 200 == 0:
        from utils.checkpoints import save_weights
        params = ll.get_all_params(dis_layers+[classifier,]+gen_layers+disxz_layers+inf_layers)
        save_weights(os.path.join(outfolder, 'model_epoch' + str(epoch) + '.npy'), params, None)
        save_weights(os.path.join(outfolder, 'average'+ str(epoch) +'.npy'), cla_param_avg, None)
