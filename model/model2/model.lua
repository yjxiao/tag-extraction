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

model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize[1], 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize[1], 1, stridesize[1], 1))

-- stage 2: 
model:add(nn.Reshape(viewsize*nstates[1]))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[1]*viewsize, nstates[2]))
model:add(nn.ReLU())

-- stage 3:
model:add(nn.Linear(nstates[2], noutputs))
model:add(nn.Sigmoid())

-- loss:
criterion = nn.PWECriterion()

