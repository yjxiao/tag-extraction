require 'torch'
require 'cunn'
require 'optim'
require 'xlua'

ffi = require('ffi')

--- Parses and loads the GloVe word vectors into a hash table:
-- glove_table['word'] = vector
function load_glove(path, inputDim)
    
    local glove_file = io.open(path)
    local glove_table = {}

    local line = glove_file:read("*l")
    while line do
        -- read the GloVe text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    glove_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                glove_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = glove_file:read("*l")
    end
    
    return glove_table
end

function preprocess_data(content, meta, wordvector_table, labelvector_table, opt)
    
    local data = torch.zeros(opt.minibatchSize, 1, opt.inputDim, opt.inputLen)
    local labels = torch.zeros(opt.minibatchSize, opt.nClasses)
 
    for i=1, opt.minibatchSize do

        -- read meta info of the post title and body
        local index = meta.index[i][1]
	local length = meta.length[i][1]

	-- standardize to all lowercase
        local document = ffi.string(torch.data(content:narrow(1, index, length))):lower()
            
	-- break each review into words and concatenate the vectors
	local doc_size = 1
        for word in document:gmatch("%S+") do
            if wordvector_table[word:gsub("%p+", "")] then
	        data[{i, 1, {}, {doc_size}}] = wordvector_table[word:gsub("%p+", "")]
            end
            if doc_size == opt.inputLen then
	        break
	    end
	    doc_size = doc_size + 1
	    
        end

	-- index and length for the labels
        index = meta.index[i][2]
	length = meta.length[i][2]

        local labelset = ffi.string(torch.data(content:narrow(1, index, length)))
        for label in labelset:gmatch("%S+") do
            labels[i]:add(labelvector_table[label])
        end
    end

    return data, labels
end

function train_model(model, criterion, data, wordvector_table, labelvector_table, opt)

    model:cuda()
    criterion:cuda()
    local cudabatch = torch.zeros(opt.minibatchSize, 1, opt.inputDim, opt.inputLen):cuda()
    local cudabatch_labels = torch.zeros(opt.minibatchSize, opt.nClasses):cuda()
    parameters, grad_parameters = model:getParameters()
    
    -- store index and length info for the current batch
    local batch_meta = {}
    batch_meta.index = torch.Tensor(opt.minibatchSize, 2)
    batch_meta.length = torch.Tensor(opt.minibatchSize, 2)

    -- optimization functional to train the model with torch's optim library
    local function feval(x)
        cudabatch[{}], cudabatch_labels[{}] = preprocess_data(data.content, batch_meta, wordvector_table, labelvector_table, opt)
        
        model:training()
	local output = model:forward(cudabatch)
        local minibatch_loss = criterion:forward(output, cudabatch_labels)
        model:zeroGradParameters()
        model:backward(cudabatch, criterion:backward(output, cudabatch_labels))
        
        return minibatch_loss, grad_parameters
    end
    
    for epoch=1,opt.nEpochs do
        local order = torch.randperm(opt.nBatches) -- not really good randomization
        for batch=1,opt.nBatches do
	    xlua.progress(batch, opt.nBatches)
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
	    batch_meta.index = data.tr_index[{{opt.idx, opt.idx+opt.minibatchSize-1}, {}}]:clone()
	    batch_meta.length = data.tr_length[{{opt.idx, opt.idx+opt.minibatchSize-1}, {}}]:clone()
            optim.sgd(feval, parameters, opt)
        end

        --local accuracy = test_model(model, test_data, test_labels, opt)
        --print("epoch ", epoch, " error: ", accuracy)

    end
end

-- TODO: rewrite function
function test_model(model, data, labels, opt)
    
    model:evaluate()

    local pred = model:forward(data)
    local _, argmax = pred:max(2)
    local err = torch.ne(argmax:double(), labels:double()):sum() / labels:size(1)

    --local debugger = require('fb.debugger')
    --debugger.enter()

    return err
end

-- TODO: rewrite main
function main()

    -- Configuration parameters
    opt = {}
    -- change these to the appropriate data locations
    opt.glovePath = "/home/xray/courses/nlp/glove.6B.50d.txt"
    opt.dataPath = "/home/xray/courses/nlp/splitted_data.t7"
    opt.labelPath = "/home/xray/courses/nlp/label_table.t7"

    -- word vector dimensionality
    opt.inputDim = 50 
    opt.inputLen = 100

    -- nTrainDocs is the number of documents used in the training set
    opt.nTrainDocs = 2879603
    opt.nTestDocs = 720833
    opt.nClasses = 2150

    -- SGD parameters - play around with these
    opt.nEpochs = 5
    opt.minibatchSize = 128
    opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
    opt.learningRate = 0.1
    opt.learningRateDecay = 0.001
    opt.momentum = 0.1
    opt.idx = 1

    print("Loading word vectors ...")
    local glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Loading raw data ...")
    local raw_data = torch.load(opt.dataPath)

    print("Loading label table ...")        
    local label_table = torch.load(opt.labelPath)

    -- construct model:
    ninputs = 1
    nstates = {50, 50, 50}
    noutputs = opt.nClasses
    
    -- convolution layers
    filtsizeW = {5, 3, 3}       -- filter size across words
    filtsizeH = {50, 1, 1}      -- filter size across word vector dimensions
    poolsizeW = {2, 2, 3}       -- pooling size across words
    poolsizeH = {1, 1, 1}       -- pooling size across word vector dimensions

    model = nn.Sequential()

    model:add(nn.SpatialConvolutionMM(ninputs, nstates[1], filtsizeW[1], filtsizeH[1], 1, filtsizeH[1])) -- 50x96x1
    model:add(nn.ReLU())    
    model:add(nn.SpatialMaxPooling(poolsizeW[1], poolsizeH[1]))						 -- 50x48x1

    model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsizeW[2], filtsizeH[2], 1, filtsizeH[2]))-- 50x46x1
    model:add(nn.ReLU())    
    model:add(nn.SpatialMaxPooling(poolsizeW[2], poolsizeH[2]))						 -- 50x23x1

    model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsizeW[3], filtsizeH[3], 1, filtsizeH[3]))-- 50x21x1
    model:add(nn.ReLU())    
    model:add(nn.SpatialMaxPooling(poolsizeW[3], poolsizeH[3]))						 -- 50x7x1

    model:add(nn.Reshape(nstates[3]*7, true))
    model:add(nn.Linear(nstates[3]*7, noutputs))

    criterion = nn.BCECriterion()
   
    train_model(model, criterion, raw_data, glove_table, label_table, opt)

end

main()
