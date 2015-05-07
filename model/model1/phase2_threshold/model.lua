require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cunn'

noutputs = noutputs or 34
batchSize = 16

-- input dimensions
nfeats = nfeats or 50
inlength = inlength or 100
ninputs = nfeats*inlength

-- hidden units, filter sizes (for ConvNet only):
nstates = {128,400}
filtsize = {21}
poolsize = {2}
stridesize = {2}
viewsize = 40

print '==> construct model'

model = torch.load('model.net')

model_thres = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model_thres:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize[1], 1))
model_thres:add(nn.ReLU())
model_thres:add(nn.SpatialMaxPooling(poolsize[1], 1, stridesize[1], 1))

-- stage 2 : 
model_thres:add(nn.Reshape(viewsize*nstates[1]))
model_thres:add(nn.Dropout(0.5))
model_thres:add(nn.Linear(nstates[1]*viewsize, nstates[2]))
model_thres:add(nn.ReLU())

-- stage 3:
model_thres:add(nn.Linear(nstates[2], 1))
model_thres:add(nn.Sigmoid())

-- loss:
criterion = nn.MSECriterion()

