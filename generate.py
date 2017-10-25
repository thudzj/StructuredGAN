'''
This code 
(1)generates data in various ways given a trained SGAN
(2)inferences latent code for test set images
(3)performs style transfer

Note: Due to the effect of Batch Normalization, it is better to generate batch_size_g (see train file, 200 for cifiar10) samples distributed equally across class in each batch.
'''
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

from layers.merge import ConvConcatLayer, MLPConcatLayer
from layers.deconv import Deconv2DLayer

from components.shortcuts import convlayer, mlplayer
from components.objectives import categorical_crossentropy_ssl_separated, maximum_mean_discripancy, categorical_crossentropy, feature_matching
from utils.create_ssl_data import create_ssl_data, create_ssl_data_subset
from utils.others import get_nonlin_list, get_pad_list, bernoullisample, printarray_2D, array2file_2D
import utils.paramgraphics as paramgraphics
from utils.checkpoints import load_weights
from visualize import *

# global
parser = argparse.ArgumentParser()
parser.add_argument("-oldmodel", type=str, default=argparse.SUPPRESS)
parser.add_argument("-dataset", type=str, default='cifar10')
args = parser.parse_args()

filename_script=os.path.basename(os.path.realpath(__file__))
outfolder='imags/cifar10_sgan/'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

# seeds
seed=1100
rng=np.random.RandomState(seed)
theano_rng=MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# G
n_z=100
batch_size_g=100
num_x=50000
# data dependent
if args.dataset == 'svhn' or args.dataset == 'cifar10':
    gen_final_non=ln.tanh
    num_classes=10
    dim_input=(32,32)
    in_channels=3
    colorImg=True
    generation_scale=True
    if args.dataset == 'svhn':
        def rescale(mat):
            return np.transpose(np.cast[theano.config.floatX]((-127.5 + mat)/127.5),(3,2,0,1))
        import svhn_data
        eval_x, eval_y = svhn_data.load('./svhn/','test')
        eval_y = np.int32(eval_y)
        eval_x = rescale(eval_x)

    else:
        def rescale(mat):
            return np.cast[theano.config.floatX](mat)
        import cifar10_data
        eval_x, eval_y = cifar10_data.load('./cifar10/','test')
        eval_y = np.int32(eval_y)
        eval_x = rescale(eval_x)

elif args.dataset == 'mnist':
    gen_final_non=ln.sigmoid
    num_classes=10
    dim_input=(28,28)
    in_channels=1
    colorImg=False
    generation_scale=False

'''
models
'''
# symbols
sym_y_g = T.ivector()
sym_z_input = T.matrix()
sym_z_rand = theano_rng.uniform(size=(batch_size_g, n_z))
sym_z_shared = T.tile(theano_rng.uniform((batch_size_g/num_classes, n_z)), (num_classes, 1))
sym_x_u_i = T.tensor4()

# generator y2x: p_g(x, y) = p(y) p_g(x | y) where x = G(z, y), z follows p_g(z)
gen_in_z = ll.InputLayer(shape=(None, n_z))
gen_in_y = ll.InputLayer(shape=(None,))
gen_layers = [gen_in_z]
if args.dataset == 'svhn' or args.dataset == 'cifar10':
    gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-00'))
    gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu, name='gen-01'), g=None, name='gen-02'))
    gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (-1,512,4,4), name='gen-03'))
    gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-10'))
    gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='gen-11'), g=None, name='gen-12')) # 4 -> 8
    gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-20'))
    gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='gen-21'), g=None, name='gen-22')) # 8 -> 16
    gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-30'))
    gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (None,3,32,32), (5,5), W=Normal(0.05), nonlinearity=gen_final_non, name='gen-31'), train_g=True, init_stdv=0.1, name='gen-32')) # 16 -> 32
elif args.dataset == 'mnist':
    gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-1'))
    gen_layers.append(ll.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=ln.softplus, name='gen-2'), name='gen-3'))
    gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-4'))
    gen_layers.append(ll.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=ln.softplus, name='gen-5'), name='gen-6'))
    gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-7'))
    gen_layers.append(nn.l2normalize(ll.DenseLayer(gen_layers[-1], num_units=28**2, nonlinearity=gen_final_non, name='gen-8')))

# inference module
inf_in_x = ll.InputLayer(shape=(None, in_channels) + dim_input)
inf_layers = [inf_in_x]
inf_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(inf_layers[-1], 64, (4,4), stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='inf-02'), name='inf-03'))
inf_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(inf_layers[-1], 128, (4,4), stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='inf-11'), name='inf-12'))
inf_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(inf_layers[-1], 256, (4,4), stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='inf-21'), name='inf-22'))
inf_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(inf_layers[-1], 512, (4,4), stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='inf-31'), name='inf-32'))
inf_layers.append(ll.DenseLayer(inf_layers[-1], num_units=n_z, W=Normal(0.05), nonlinearity=None, name='inf-4'))

# outputs
gen_out_x = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_rand}, deterministic=False)
gen_out_x_shared = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_shared}, deterministic=False)
gen_out_x_interpolation = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_input}, deterministic=False)
generate = theano.function(inputs=[sym_y_g], outputs=gen_out_x)
generate_shared = theano.function(inputs=[sym_y_g], outputs=gen_out_x_shared)
generate_interpolation = theano.function(inputs=[sym_y_g, sym_z_input], outputs=gen_out_x_interpolation)
inf_z = ll.get_output(inf_layers[-1], {inf_in_x:sym_x_u_i}, deterministic=False)
inf = theano.function(inputs=[sym_x_u_i], outputs=inf_z)

'''
Load pretrained model
'''
load_weights(args.oldmodel, gen_layers+inf_layers)

# style transfer
# batch_size=100
# features = np.load("trans.npy").astype(np.float32)#np.zeros([100, n_z])#
# for ppp in range(10):
#     images = np.zeros([10*32, 11*32 + 3, 3])
#     eval_x = eval_x[rng.permutation(eval_x.shape[0])]
#     for t in range(10):
#         train_xu = eval_x[t*batch_size:(t+1)*batch_size]
        
#         images[t*32:(t+1)*32, 0:32, :] = np.transpose(train_xu[0], [1, 2, 0])
#         z_ = inf(train_xu)
#         features[t*10:(t+1)*10] = z_[:10]
#         #z_ = features[t*10:(t+1)*10]

#         print(sum(z_[0]*z_[0]))
#         sample_y = np.int32(np.repeat(np.arange(10), batch_size/10))
#         images_ = generate_interpolation(sample_y, np.tile(z_[:10], [10, 1])) #sess.run(x_gen, feed_dict={y_in:sample_y, z_in:np.tile(z_[:10], [10, 1])})
#         for j in range(10):
#             images[t*32:(t+1)*32, 35+j*32:35+(j+1)*32, :] = np.transpose(images_[j*10], [1, 2, 0])
#     scipy.misc.imsave(os.path.join(outfolder, 'tran-'+args.dataset+"-"+str(ppp)+'.png'), images)
#np.save("trans", features)


# generate images in various ways
images = [[] for i in range(10)]
for i in xrange(10):
    sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    # inds = np.random.permutation(batch_size_g)
    # sample_y = sample_y[inds]
    x_gen_batch = generate(sample_y)
    for j in range(10):
        images[j].append(x_gen_batch[j*10:(j+1)*10])
    #x_gen_batch = x_gen_batch.reshape((batch_size_g,-1))
    #scipy.misc.imsave(os.path.join(outfolder, 'class-'+str(i)+'.png'), concat_multiple_images(np.transpose((x_gen_batch[:100]+1)/2, [0,2,3,1])))
for i in range(10):
    tmp = np.concatenate(images[i], 0)
    scipy.misc.imsave(os.path.join(outfolder, 'class-'+str(i)+'.png'), concat_multiple_images(np.transpose((tmp+1)/2, [0,2,3,1])))
# exit(0)

#interpolation on latent space (z) class conditionally
for i in xrange(10):
    sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    orignial_z = np.repeat(rng.uniform(size=(num_classes,n_z)), batch_size_g/num_classes, axis=0)
    target_z = np.repeat(rng.uniform(size=(num_classes,n_z)), batch_size_g/num_classes, axis=0)
    alpha = np.tile(np.arange(batch_size_g/num_classes) * 1.0 / (batch_size_g/num_classes-1), num_classes)
    alpha = alpha.reshape(-1,1)
    z = np.float32((1-alpha)*orignial_z+alpha*target_z)
    x_gen_batch = generate_interpolation(sample_y, z)
    #x_gen_batch = x_gen_batch.reshape((batch_size_g,-1))
    scipy.misc.imsave(os.path.join(outfolder, 'interpolation-'+str(i)+'.png'), concat_multiple_images(np.transpose((x_gen_batch[:100]+1)/2, [0,2,3,1])))
    #image = paramgraphics.mat_to_img(x_gen_batch.T, dim_input, colorImg=colorImg, tile_shape=(num_classes, 2*num_classes), scale=generation_scale, save_path=os.path.join(outfolder, 'interpolation-'+str(i)+'.png'))

# class conditionally generation with shared z and fixed y
for i in xrange(10):
    sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    x_gen_batch = generate_shared(sample_y)
    #x_gen_batch = x_gen_batch.reshape((batch_size_g,-1))
    scipy.misc.imsave(os.path.join(outfolder, 'shared-'+str(i)+'.png'), concat_multiple_images(np.transpose((x_gen_batch[:100]+1)/2, [0,2,3,1])))
    #image = paramgraphics.mat_to_img(x_gen_batch.T, dim_input, colorImg=colorImg, tile_shape=(num_classes, 2*num_classes), scale=generation_scale, save_path=os.path.join(outfolder, 'shared-'+str(i)+'.png'))

# generation with randomly sampled z and y
for i in xrange(10):
    sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    # inds = np.random.permutation(batch_size_g)
    # sample_y = sample_y[inds]
    x_gen_batch = generate(sample_y)
    #x_gen_batch = x_gen_batch.reshape((batch_size_g,-1))
    scipy.misc.imsave(os.path.join(outfolder, 'random-'+str(i)+'.png'), concat_multiple_images(np.transpose((x_gen_batch[:100]+1)/2, [0,2,3,1])))
    #image = paramgraphics.mat_to_img(x_gen_batch.T, dim_input, colorImg=colorImg, tile_shape=(num_classes, 2*num_classes), scale=generation_scale, save_path=os.path.join(outfolder, 'random-'+str(i)+'.png'))

#randomly sample real images
for i in range(10):
    bx = eval_x[rng.permutation(eval_x.shape[0])][:100]
    scipy.misc.imsave(os.path.join(outfolder, 'sample-'+str(i)+'.png'), concat_multiple_images(np.transpose((bx+1)/2, [0,2,3,1])))

#randomly sample real images condition on class
for i in range(10):
    inds = np.random.permutation(eval_x.shape[0])
    eval_y = eval_y[inds]
    eval_x = eval_x[inds]
    bx = []
    for j in range(10):
        bx.append(eval_x[eval_y==j][:10])
    #bx = eval_x[rng.permutation(eval_x.shape[0])][:100]
    bx = np.concatenate(bx, 0)
    #bx = np.reshape(np.transpose(np.reshape(bx, [10, 10, 3, 32, 32]), [1,0,2,3,4]), [100, 3, 32, 32])
    scipy.misc.imsave(os.path.join(outfolder, 'sample-cond-'+str(i)+'.png'), concat_multiple_images(np.transpose((bx+1)/2, [0,2,3,1])))


# randomly generate 50000 images for inception score computation
gens = 50000
images = np.zeros([gens, 3, 32, 32])
labels = np.zeros([gens])
for t in range(int(gens/batch_size_g)):
    #sample_y = np.argmax(np.random.multinomial(1, [1/10.]*10, size=batch_size_g), 1)#np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    inds = np.random.permutation(batch_size_g)
    sample_y = sample_y[inds]
    image_ = generate(sample_y)
    images[t*batch_size_g:(t+1)*batch_size_g] = image_#np.reshape(image_, [batch_size_g, 28, 28, 1])
    labels[t*batch_size_g:(t+1)*batch_size_g] = sample_y
print(labels[100:200])
np.savez('generations-sgan-inc-50000.npz', x=images[:gens], y=labels[:gens])

#  inference latent code for test images
features = np.zeros([eval_x.shape[0], 100])
print(eval_y.shape)
labels = eval_y
for i in range(int(math.ceil(eval_x.shape[0]/100.0))):
    inf_z_ = inf(eval_x[100*i:min(100*(i+1), eval_x.shape[0])])
    features[100*i:min(100*(i+1), eval_x.shape[0])] = inf_z_
print(features.shape)
np.savez('features-sgan-cifar.npz', x=features, y=labels)

