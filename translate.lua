require 'torch'
require 'cunn'
require 'optim'

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

function preprocess_data(raw_data, meta, wordvector_table, labelvector_table, opt)
    
    local data = torch.zeros(opt.minibatchSize, 1, opt.inputDim, opt.inputLen)
    local labels = torch.zeros(opt.minibatchSize, opt.nClasses)
 
    for i=1, opt.minibatchSize do

        -- read meta info of the post title and body
        local index = meta.index[i][1]
	local length = meta.length[i][1]

	-- standardize to all lowercase
        local document = ffi.string(torch.data(raw_data.content:narrow(1, index, length))):lower()
            
	-- break each review into words and compute the document average
	local doc_size = 1
        for word in document:gmatch("%S+") do
            if wordvector_table[word:gsub("%p+", "")] then
	        data[{i, 1, {}, {doc_size}}] = wordvector_table[word:gsub("%p+", "")]
            end
	    doc_size = doc_size + 1
        end

	-- index and length for the labels
        index = meta.index[i][2]
	length = meta.length[i][2]

        local labelset = ffi.string(torch.data(raw_data.content:narrow(1, index, length)))
	local doc_size = 1
        for label in labelset:gmatch("%S+") do
            labels[i]:add(labelvector_table[label])
            doc_size = doc_size + 1
        end
    end

    return data, labels
end

function train_model(model, criterion, raw_data, meta, wordvector_table, labelvector_table, opt)

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
        cudabatch[{}], cudabatch_labels[{}] = preprocess_data(raw_data, batch_meta, wordvector_table, labelvector_table, opt)
        
        model:training()
        local minibatch_loss = criterion:forward(model:forward(cudabatch), cudabatch_labels)
        model:zeroGradParameters()
        model:backward(cudabatch, criterion:backward(model.output, cudabatch_labels))
        
        return minibatch_loss, grad_parameters
    end
    
    for epoch=1,opt.nEpochs do
        local order = torch.randperm(opt.nBatches) -- not really good randomization
        for batch=1,opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
	    batch_meta.index = meta.index[{{opt.idx, opt.idx+opt.minibatchSize-1}, {}}]:clone()
	    batch_meta.length = meta.length[{{opt.idx, opt.idx+opt.minibatchSize-1}, {}}]:clone()
            optim.sgd(feval, parameters, opt)
            print("epoch: ", epoch, " batch: ", batch)
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
    opt.glovePath = "CHANGE_ME" -- path to raw glove data .txt file
    opt.dataPath = "CHANGE_ME"
    -- word vector dimensionality
    opt.inputDim = 50 
    -- nTrainDocs is the number of documents per class used in the training set, i.e.
    -- here we take the first nTrainDocs documents from each class as training samples
    -- and use the rest as a validation set.
    opt.nTrainDocs = 10000
    opt.nTestDocs = 0
    opt.nClasses = 5
    -- SGD parameters - play around with these
    opt.nEpochs = 5
    opt.minibatchSize = 128
    opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
    opt.learningRate = 0.1
    opt.learningRateDecay = 0.001
    opt.momentum = 0.1
    opt.idx = 1

    print("Loading word vectors...")
    local glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Loading raw data...")
    local raw_data = torch.load(opt.dataPath)
    
    print("Computing document input representations...")
    local processed_data, labels = preprocess_data(raw_data, glove_table, opt)
    
    -- split data into makeshift training and validation sets
    local training_data = processed_data:sub(1, opt.nClasses*opt.nTrainDocs, 1, processed_data:size(2)):clone()
    local training_labels = labels:sub(1, opt.nClasses*opt.nTrainDocs):clone()
    
    -- make your own choices - here I have not created a separate test set
    local test_data = training_data:clone() 
    local test_labels = training_labels:clone()

    -- construct model:
    model = nn.Sequential()
   
    model:add(nn.TemporalConvolution(1, 20, 10, 1))
    model:add(nn.TemporalMaxPooling(3, 1))
    
    model:add(nn.Reshape(20*39, true))
    model:add(nn.Linear(20*39, 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()
   
    train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
    local results = test_model(model, test_data, test_labels)
    print(results)
end

--main()
