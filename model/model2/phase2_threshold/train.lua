require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

print '==> configuring optimizer'


save = '.'
maxEpoch = 500

optimState = {
   learningRate = 0.01,
   momentum = 0.9,
   learningRateDecay = 0.000005
}
optimMethod = optim.sgd

-- Log results to files
trainLogger = optim.Logger(paths.concat(save, 'train.log'))
testLogger = optim.Logger(paths.concat(save, 'test.log'))

model:float()
model_thres:float()
criterion:float()

if opt.type == 'cuda' then 
   model:cuda()
   model_thres:cuda()
   criterion:cuda()
end

print '==> defining training procedure'
parameters, gradParameters = model:getParameters()

function get_rank_loss(output, target)
   local pos = {}
   local neg = {}
   local loss = 0
   for k = 1, target:size(1) do
      if target[k] >= 1 then
	 table.insert(pos, k)
      else
	 table.insert(neg, k)
      end
   end
   for k = 1, #pos do
      for kk = 1, #neg do
	 if output[pos[k]] <= output[neg[kk]] then
	    loss = loss + 1
	 end
      end
   end
   loss = loss / #pos / #neg
   return loss
end

function getOptimalThreshold(input, target)
   local res = input.new()
   res:resize(1)
   input = input:float()
   target = target:float()
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
   res[1] = best_thres
   return res
end


function train()
   shuffle = torch.randperm(trainData.size)

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   local tloss = 0
   local correct = 0
   local n_correct = 0
   local exact_correct = 0
   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,trainData.size,batchSize do
      -- disp progress
      xlua.progress(t, trainData.size)

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,trainData.size) do
         -- load new sample
	 local input
	 local target
         input, target = preprocess_data(all_data.content, trainData.index[shuffle[i]],
				       trainData.length[shuffle[i]], glove_table, label_table)
         if opt.type == 'cuda' then 
	    input = input:cuda() 
	    target = target:cuda()
	 end
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
			  local target_thres = getOptimalThreshold(output, targets[i])
			  local output_thres = model_thres:forward(inputs[i])
                          local err = criterion:forward(output_thres, target_thres)
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output_thres, target_thres)
                          model_thres:backward(inputs[i], df_do)

                          -- update confusion
			  local temp = output:ge(output_thres[1]):eq(targets[i]:ge(0.5))
			  correct = correct + temp:sum()
			  if temp:sum() == noutputs then
			     exact_correct = exact_correct + 1
			  end
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
		       tloss = tloss + f
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
      end
      
      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
      
   end
   -- time taken
   tloss = tloss / trainData.size
   time = sys.clock() - time
   time = time / trainData.size

   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   print("\n==> training accuracy % under predicted threshold:")
   print(correct / trainData.size / noutputs * 100)
   print("\n==> training exact match accuracy % under predicted thres:")
   print(exact_correct / trainData.size * 100)
   print("\n==> training loss for predicting thres")
   print(tloss)

   -- update logger/plot
   trainLogger:add{['% class accuracy (train set)'] = correct / trainData.size / noutputs * 100,
      ['training loss'] = tloss, 
      ['training exact match accuracy %'] = exact_correct / trainData.size * 100}

   -- save/log current net
   local filename = paths.concat(save, 'model_thres.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

print('==> defining test procedure')

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')

   local tloss = 0
   local correct = 0
   local exact_correct = 0
   local n_tp = 0
   local n_pred = 0
   local n_pos = 0
   local f_mac_tp = torch.Tensor(noutputs):zero():float()
   local f_mac_pred = torch.Tensor(noutputs):zero():float()
   local f_mac_pos = torch.Tensor(noutputs):zero():float()

      -- disp progress
   for t = 1,testData.size do
      -- disp progress
      xlua.progress(t, testData.size)
      local input
      local target
      input, target = preprocess_data(all_data.content, testData.index[t],
				      testData.length[t], glove_table, label_table)
      if opt.type == 'cuda' then
        input = input:cuda()
        target = target:cuda()
      end
      -- test sample
      local pred = model:forward(input)
      local target_thres = getOptimalThreshold(pred, target)
      local pred_thres = model_thres:forward(input)
      local loss = criterion:forward(pred_thres, target_thres)
      tloss = tloss + loss
      local temp = pred:ge(pred_thres[1]):eq(target:ge(0.5))
      correct = correct + temp:sum()
      if temp:sum() == noutputs then
	 exact_correct = exact_correct + 1
      end
      f_mac_tp:add(pred:ge(pred_thres[1]):cmul(target:ge(0.5)):float())
      f_mac_pos:add(target:ge(0.5):float())
      f_mac_pred:add(pred:ge(pred_thres[1]):float())
      n_tp = n_tp + pred:ge(pred_thres[1]):cmul(target:ge(0.5)):sum()
      n_pred = n_pred + pred:ge(pred_thres[1]):sum()
      n_pos = n_pos + target:ge(0.5):sum()
      -- a more agressive modified accuracy as follows:
      -- n_correct = n_correct + present:sum() - pred:ge(0.5):eq(target:le(0.5)):sum()
      -- print("\n" .. target .. "\n")

   end
   
   tloss = tloss / testData.size
   -- timing
   time = sys.clock() - time
   time = time / testData.size
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print('\n Test Accuracy % under predicted thres:')
   print(correct / testData.size / noutputs * 100)
   print('\ntest exact match accuracy % under predicted thres:')
   print(exact_correct / testData.size * 100)
   print('\ntest loss for predicting thres:')
   print(tloss)
   print('\ntest precision under predicted thres:')
   print(n_tp / n_pred)
   print('\ntest recall under predicted thres:')
   print(n_tp / n_pos)
   print('\ntest F micro under predicted thres')
   print(2 * n_tp / (n_pred + n_pos))
   print('\ntest F macro under predicted thres')
   f_mac_pos:add(f_mac_pred)
   local tt = 0
   local cnt = 0
   for i=1,(#f_mac_tp)[1] do
      if f_mac_pos[i] > 0 then
         cnt = cnt + 1
         tt = tt + f_mac_tp[i] / f_mac_pos[i]
      end
   end
   print(tt * 2 / cnt)
   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = correct / testData.size / noutputs * 100,
      ['test loss'] = tloss, 
      ['test exact match accuracy %'] = exact_correct / testData.size * 100,
      ['test precision'] = n_tp / n_pred, ['test recall'] = n_tp / n_pos,
      ['test F micro'] = 2 * n_tp / (n_pred + n_pos), ['test F macro'] = (tt * 2 / cnt)}   
   -- next iteration:

end


