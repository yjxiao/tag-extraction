require 'torch'

function transform_labels(path, nClasses)
    
    local label_file = io.open(path)
    local label_table = {}

    local line = label_file:read("*l")
    local idx = 1
    while line do
	local label = line:gsub("\n", "")
	-- perform one-hot encoding for the labels
	label_table[label] = torch.zeros(nClasses):float()
	label_table[label][idx] = 1
	idx = idx + 1
	line = label_file:read("*l")
    end

    return label_table
end

label_path = "/home/xray/courses/nlp/labels.txt"
n_labels = 2150
label_table = transform_labels(label_path, n_labels)

torch.save("/home/xray/courses/nlp/label_table.t7", label_table)
