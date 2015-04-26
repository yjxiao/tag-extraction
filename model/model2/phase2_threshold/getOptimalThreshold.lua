require 'torch'

function getOptimalThreshold(input, target)
   local y
   local idx
   y, idx = torch.sort(input, 1, true)
   local best_thres = y[1]
   local best_f = 0
   for k = 1, input:size(1) - 1 do
      local threshold = (y[k] + y[k+1]) / 2
      local pred = input:ge(threshold):float()
      local f_m = 2 * pred:clone():cmul(target):sum() / (pred:sum() + target:sum())
      if f_m > best_f then
	 best_f = f_m
	 best_thres = threshold
      end
   end
   return best_thres
end
