# given two datasets merge them together into one dataset
# when merging every image in the 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import mxnet as mx
from mxnet import ndarray as nd
import random
import argparse
import cv2
import time
import sklearn
import numpy as np
import csv

sys.path.append(os.path.join(os.path.dirname(__file__),'..', 'common'))
import face_image

sys.path.append(os.path.join(os.path.dirname(__file__),'..', 'eval'))
import verification

# use the given to get the embedding vectors for given images
# the returned verctors are normalized  
def get_embedding(args, imgrec, id, image_size, model):
  s = imgrec.read_idx(id)
  header, _ = mx.recordio.unpack(s)
  ocontents = []
  # print('** get_embedding:', int(header.label[0]), int(header.label[1]))
  # for idx in range(int(imgrec.keys[0]), int(imgrec.keys[-1])):
  # for idx in range(int(id), int(id)+1):
  upper_image_index = min(args.max_image_per_identity_in_embedding + int(header.label[0]), int(header.label[1]))
  for idx in xrange(int(header.label[0]), upper_image_index):
    # print('** get_embedding idx', idx)
    s = imgrec.read_idx(idx)
    ocontents.append(s)
  embeddings = None
  # print('len(ocontents)', len(ocontents))
  ba = 0
  while True:
    bb = min(ba+args.batch_size, len(ocontents))
    if ba>=bb:
      break
    _batch_size = bb-ba
    _batch_size2 = max(_batch_size, args.ctx_num)
    data = nd.zeros( (_batch_size2,3, image_size[0], image_size[1]) )
    #label = nd.zeros( (_batch_size2,) )
    count = bb-ba
    ii=0
    for i in xrange(ba, bb):
      header, img = mx.recordio.unpack(ocontents[i])
      #print(header.label.shape, header.label)
      img = mx.image.imdecode(img)
      img = nd.transpose(img, axes=(2, 0, 1))
      data[ii][:] = img
      #label[ii][:] = header.label
      ii+=1
    while ii<_batch_size2:
      data[ii][:] = data[0][:]
      #label[ii][:] = label[0][:]
      ii+=1
    #db = mx.io.DataBatch(data=(data,), label=(label,))
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    net_out = model.get_outputs()
    net_out = net_out[0].asnumpy()
    if embeddings is None:
      embeddings = np.zeros( (len(ocontents), net_out.shape[1]))
    embeddings[ba:bb,:] = net_out[0:_batch_size,:]
    ba = bb
  embeddings = sklearn.preprocessing.normalize(embeddings)
  # print('get_embedding:embeddings.shape:', embeddings.shape)
  embedding = np.mean(embeddings, axis=0, keepdims=True)
  # print('get_embedding:embedding.shape:', embedding.shape)
  embedding = sklearn.preprocessing.normalize(embedding).flatten()
  # print('get_embedding:embedding.shape:', embedding.shape)
  # print('get_embedding:', embedding)
  return embedding

def main(args):
  include_datasets = args.include.split(',')
  prop = face_image.load_property(include_datasets[0])
  image_size = prop.image_size
  print('image_size', image_size)
  model = None
  if len(args.model)>0:
    ctx = []
    cvd = ''
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
      cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in xrange(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))
    args.ctx_num = len(ctx)
    vec = args.model.split(',')
    prefix = vec[0]
    epoch = int(vec[1])
    print('loading',prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    #model = mx.mod.Module.load(prefix, epoch, context = ctx)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)

  rec_list = []
  for ds in include_datasets:
    path_imgrec = os.path.join(ds, 'train.rec')
    path_imgidx = os.path.join(ds, 'train.idx')
    # allow using user paths by expanding tilde
    path_imgrec = os.path.expanduser(path_imgrec)
    path_imgidx = os.path.expanduser(path_imgidx)
    # print('path_imgrec:', path_imgrec)
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
    # print('imgrec:', imgrec)
    rec_list.append(imgrec)
  id_list_map = {}
  all_id_list = []
  test_limit = 0
  for ds_id in xrange(len(rec_list)):
    id_list = []
    imgrec = rec_list[ds_id]
    # print('----',ds_id,'----')
    # print('imgrec:', imgrec)
    # print('keys:', imgrec.keys)
    s = imgrec.read_idx(0)
    # s = imgrec.read()
    header, _ = mx.recordio.unpack(s)
    # print('header:', header)
    assert header.flag>0
    print('header0 label', header.label)
    header0 = (int(header.label[0]), int(header.label[1]))
    #assert(header.flag==1)
    imgidx = range(1, int(header.label[0]))
    id2range = {}
    # print('** get_embedding:', int(header.label[0]), int(header.label[1]))
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    pp=0
    for identity in seq_identity:
      pp+=1
      if pp%10==0:
        print('processing id', pp)
      if model is not None:
        embedding = get_embedding(args, imgrec, identity, image_size, model)
      else:
        embedding = None
      #print(embedding.shape)
      id_list.append( [ds_id, identity, embedding] )
      if test_limit>0 and pp>=test_limit:
        break
    id_list_map[ds_id] = id_list
    if ds_id==0 or model is None:
      all_id_list += id_list
      print(ds_id, len(id_list))
    else:
      X = []
      for id_item in all_id_list:
        X.append(id_item[2])
      X = np.array(X)
      for i in xrange(len(id_list)):
        id_item = id_list[i]
        y = id_item[2]
        sim = np.dot(X, y.T)
        # print('sim:', sim)
        idx = np.where(sim>=args.similarity_upper_threshold_include)[0]
        # we are confident this identity already exists in the set
        if len(idx)>0:
          continue
        idx = np.where(sim>=args.similarity_lower_threshold_include)[0]
        # this identity might not exist in the set, let's manually check that
        line = []
        if len(idx)>0:
          # print('possible duplicate:', idx)
          # store in the file the current set path, current identity, [possible duplicate set path, possible duplicate identity]xn
          # where n is number of possible duplicates found
          line.append(include_datasets[ds_id])
          line.append(id_item[1])
          for duplicate_id in idx:
            duplicate_dataset_index = all_id_list[duplicate_id][0]
            line.append(include_datasets[duplicate_dataset_index])
            duplicate_identity = all_id_list[duplicate_id][1]
            line.append(duplicate_identity)
          with open('duplicates.csv', 'a') as csv_file:
              writer = csv.writer(csv_file, delimiter = ',')
              # print('line: ',line)
              writer.writerows([line])
          continue
        all_id_list.append(id_item)
      print(ds_id, len(id_list), len(all_id_list))


  if len(args.exclude)>0:
    if os.path.isdir(args.exclude):
      _path_imgrec = os.path.join(args.exclude, 'train.rec')
      _path_imgidx = os.path.join(args.exclude, 'train.idx')
      _imgrec = mx.recordio.MXIndexedRecordIO(_path_imgidx, _path_imgrec, 'r')  # pylint: disable=redefined-variable-type
      _ds_id = len(rec_list)
      _id_list = []
      s = _imgrec.read_idx(0)
      header, _ = mx.recordio.unpack(s)
      assert header.flag>0
      print('header0 label', header.label)
      header0 = (int(header.label[0]), int(header.label[1]))
      #assert(header.flag==1)
      imgidx = range(1, int(header.label[0]))
      seq_identity = range(int(header.label[0]), int(header.label[1]))
      pp=0
      for identity in seq_identity:
        pp+=1
        if pp%10==0:
          print('processing ex id', pp)
        embedding = get_embedding(args, _imgrec, identity, image_size, model)
        #print(embedding.shape)
        _id_list.append( (_ds_id, identity, embedding) )
        if test_limit>0 and pp>=test_limit:
          break
    else:
      _id_list = []
      data_set = verification.load_bin(args.exclude, image_size)[0][0]
      print(data_set.shape)
      data = nd.zeros( (1,3,image_size[0], image_size[1]))
      for i in range(data_set.shape[0]):
        data[0] = data_set[i]
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        net_out = model.get_outputs()
        embedding = net_out[0].asnumpy().flatten()
        _norm=np.linalg.norm(embedding)
        embedding /= _norm
        _id_list.append( (i, i, embedding) )

      X = []
      for id_item in all_id_list:
        X.append(id_item[2])
      X = np.array(X)
      emap = {}
      for id_item in _id_list:
        y = id_item[2]
        sim = np.dot(X, y.T)
        idx = np.where(sim>=args.similarity_threshold_exclude)[0]
        for j in idx:
          emap[j] = 1
          all_id_list[j][1] = -1
      print('exclude', len(emap))

  if args.test>0:
    return

  if not os.path.exists(args.output):
    os.makedirs(args.output)
  writer = mx.recordio.MXIndexedRecordIO(os.path.join(args.output, 'train.idx'), os.path.join(args.output, 'train.rec'), 'w')
  idx = 1
  identities = []
  nlabel = -1
  for id_item in all_id_list:
    if id_item[1]<0:
      continue
    nlabel+=1
    ds_id = id_item[0]
    imgrec = rec_list[ds_id]
    id = id_item[1]
    s = imgrec.read_idx(id)
    header, _ = mx.recordio.unpack(s)
    a, b = int(header.label[0]), int(header.label[1])
    identities.append( (idx, idx+b-a) )
    for _idx in xrange(a,b):
      s = imgrec.read_idx(_idx)
      _header, _content = mx.recordio.unpack(s)
      nheader = mx.recordio.IRHeader(0, nlabel, idx, 0)
      s = mx.recordio.pack(nheader, _content)
      writer.write_idx(idx, s)
      idx+=1
  id_idx = idx
  for id_label in identities:
    _header = mx.recordio.IRHeader(1, id_label, idx, 0)
    s = mx.recordio.pack(_header, '')
    writer.write_idx(idx, s)
    idx+=1
  _header = mx.recordio.IRHeader(1, (id_idx, idx), 0, 0)
  s = mx.recordio.pack(_header, '')
  writer.write_idx(0, s)
  with open(os.path.join(args.output, 'property'), 'w') as f:
    f.write("%d,%d,%d"%(len(identities), image_size[0], image_size[1]))

if __name__ == '__main__':
  try:
    os.remove('duplicates.csv')
  except OSError:
    pass

  parser = argparse.ArgumentParser(description='do dataset merge')
  # general
  parser.add_argument('--include', default='', type=str, help='')
  parser.add_argument('--exclude', default='', type=str, help='')
  parser.add_argument('--output', default='', type=str, help='')
  parser.add_argument('--model', default='../model/softmax,50', help='path to load model.')
  parser.add_argument('--batch-size', default=32, type=int, help='')
  parser.add_argument('--similarity_lower_threshold_include', default=0.3, type=float, help='any pair with similarity lower than this threshold is considered not a match.')
  parser.add_argument('--similarity_upper_threshold_include', default=0.7, type=float, help='any pair with similarity higher than this threshold is considered a match.')
  parser.add_argument('--max_image_per_identity_in_embedding', default=10, type=int, help='the cap on the number of images used per identity to compute the average embedding vector.')
  parser.add_argument('--similarity_threshold_exclude', default=0.4, type=float, help='')
  parser.add_argument('--mode', default=1, type=int, help='')
  parser.add_argument('--test', default=0, type=int, help='')
  args = parser.parse_args()
  main(args)

