require 'torch'
require 'xlua'

ffi = require('ffi')

function reformat(raw_data, opt)
    
    local f = io.open("train.tsv", "w") 
    print("Writing training data ...")
    for i=1, opt.nTrainDocs do
        xlua.progress(i, opt.nTrainDocs)
        -- read meta info of the post title and body
        local index = raw_data.tr_index[i][1]
	local length = raw_data.tr_length[i][1]

	-- standardize to all lowercase
        local document = ffi.string(torch.data(raw_data.content:narrow(1, index, length))):lower()
	f:write(document .. "\t")

	-- index and length for the labels
        index = raw_data.tr_index[i][2]
	length = raw_data.tr_length[i][2]

        local labelset = ffi.string(torch.data(raw_data.content:narrow(1, index, length)))
	f:write(labelset .. "\n")
    end

    f:close()
    print("Writing test data ...")
    f = io.open("test.tsv", "w")
    for i=1, opt.nTestDocs do
        xlua.progress(i, opt.nTestDocs)
        -- read meta info of the post title and body
        local index = raw_data.te_index[i][1]
	local length = raw_data.te_length[i][1]

	-- standardize to all lowercase
        local document = ffi.string(torch.data(raw_data.content:narrow(1, index, length))):lower()
	f:write(document .. "\t")

	-- index and length for the labels
        index = raw_data.te_index[i][2]
	length = raw_data.te_length[i][2]

        local labelset = ffi.string(torch.data(raw_data.content:narrow(1, index, length)))
	f:write(labelset .. "\n")
    end
    f:close()

end

function main()

    -- Configuration parameters
    opt = {}
    -- change these to the appropriate data locations
    opt.dataPath = "/home/xray/courses/nlp/splitted_data.t7"

    -- word vector dimensionality
    opt.inputDim = 50 
    opt.inputLen = 100

    -- nTrainDocs is the number of documents used in the training set
    opt.nTrainDocs = 2879603
    opt.nTestDocs = 720833
    opt.nClasses = 2150

    print("Loading raw data ...")
    local raw_data = torch.load(opt.dataPath)
    reformat(raw_data, opt)
    
end

main()
