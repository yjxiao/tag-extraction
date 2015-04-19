require 'torch'

function read_split(ftrain, ftest)
   local split = {}
   local train_size = 0
   local test_size = 0
   f = io.open(ftrain, 'r')
   for line in f:lines() do
      split[tonumber(line)+1] = 1   -- index start from 1
      train_size = train_size + 1
   end
   f:close()
   f = io.open(ftest, 'r')
   for line in f:lines() do
      if split[tonumber(line)+1] then
	 error('duplicate entries in train and test')
      end
      split[tonumber(line)+1] = 2   -- index start from 1
      test_size = test_size + 1
   end
   f:close()
   return split, train_size, test_size
end

data = torch.load('train.t7')
ftrain = 'train.txt'
ftest = 'test.txt'
split, train_size, test_size = read_split(ftrain, ftest)

train_index = torch.LongTensor(train_size, 2)
train_length = torch.LongTensor(train_size, 2)
test_index = torch.LongTensor(test_size, 2)
test_length = torch.LongTensor(test_size, 2)

train_cnt = 0
test_cnt = 0
for k, v in pairs(split) do
   if v == 1 then
      train_cnt = train_cnt + 1
      train_index[train_cnt] = data.index[k]
      train_length[train_cnt] = data.length[k]
   else
      test_cnt = test_cnt + 1
      test_index[test_cnt] = data.index[k]
      test_length[test_cnt] = data.length[k]
   end
end

new_data = {content = data.content, tr_index = train_index, tr_length = train_length,
	    te_index = test_index, te_length = test_length}
torch.save('splitted_data.t7', new_data)

assert(train_cnt == train_size)
assert(test_cnt == test_size)

