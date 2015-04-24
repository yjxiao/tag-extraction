require 'torch'
ffi = require('ffi')

data = torch.load('train.t7')
cnt = {}
for i = 1, data.index:size(1) do
   local labelset = ffi.string(torch.data(data.content:narrow(1, data.index[i][2], data.length[i][2])))   
   for label in labelset:gmatch("%S+") do
      cnt[label] = cnt[label] or 0
      cnt[label] = cnt[label] + 1
   end
end
print(cnt)
