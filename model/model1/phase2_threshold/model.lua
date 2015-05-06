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
nstates = {128,128,128,400}
filtsize = {21, 8, 6}
poolsize = {2, 2, 2}
stridesize = {2, 2, 2}
viewsize = 40

print '==> construct model'

model = torch.load('model.net')

model_thres = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model_thres:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize[1], 1))
model_thres:add(nn.ReLU())
model_thres:add(nn.SpatialMaxPooling(poolsize[1], 1, stridesize[1], 1))

--[[
-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model_thres:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize[2], 1))
model_thres:add(nn.ReLU())
model_thres:add(nn.SpatialMaxPooling(poolsize[2], 1, stridesize[2], 1))

-- stage 3 :
model_thres:add(nn.SpatialConvolution(nstates[2], nstates[3], filtsize[3], 1))
model_thres:add(nn.ReLU())
model_thres:add(nn.SpatialMaxPooling(poolsize[3], 1, stridesize[3], 1))
--]]

-- stage 3 : 
model_thres:add(nn.Reshape(viewsize*nstates[3]))
model_thres:add(nn.Dropout(0.5))
model_thres:add(nn.Linear(nstates[3]*viewsize, nstates[4]))
model_thres:add(nn.ReLU())

-- stage 4:
model_thres:add(nn.Linear(nstates[4], 1))
model_thres:add(nn.Sigmoid())

-- loss:
criterion = nn.MSECriterion()

