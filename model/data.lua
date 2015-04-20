require 'torch'
ffi = require('ffi')

torch.setdefaulttensortype('torch.FloatTensor')
data_path = 'splitted_data.t7'
glove_path = ''
label_path = ''

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

print '==> loading dataset'
nfeats = 50
all_data = torch.load(data_path)
print '==> loading glove vectors'
glove_table = load_glove(glove_path, nfeats)
print '==> loading labels'
label_table = torch.load(label_path)

noutputs = 2150
length = 100

trainData = {
   index = all_data.tr_index,
   length = all_data.tr_length,
   size = (#all_data.tr_index)[1]
}
testData = {
   index = all_data.te_index,
   index = all_data.te_length,
   size = (#all_data.te_index)[1]
}

function preprocess_data(content, index, length, wordvector_table, labelvector_table)
    
    local data = torch.zeros(nfeats, 1, length)
    local labels = torch.zeros(noutputs)   

    -- standardize to all lowercase
    local document = ffi.string(torch.data(content:narrow(1, index[1], length[1]))):lower()
    
    -- break each review into words and concatenate the vectors
    local doc_size = 1
    for word in document:gmatch("%S+") do
       if wordvector_table[word:gsub("%p+", "")] then
	  data[{{}, 1, {doc_size}}] = wordvector_table[word:gsub("%p+", "")]
       end
       if doc_size == length then
	  break
       end
       doc_size = doc_size + 1
    end

    local labelset = ffi.string(torch.data(content:narrow(1, index[2], length[2])))
    for label in labelset:gmatch("%S+") do
       labels:add(labelvector_table[label])
    end

    return data, labels
end

