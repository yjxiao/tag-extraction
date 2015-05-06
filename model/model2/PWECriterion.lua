require 'cunn'

local PWECriterion, parent = torch.class('nn.PWECriterion', 'nn.Criterion')

function PWECriterion:__init()
   parent.__init(self)
end 

function PWECriterion:updateOutput(input, target)
   -- BP-MLL, a convex surrogate loss for ranking loss
   -- from the paper: http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkde06a.pdf
   input = input:float()
   target = target:float()
   local pos = {}
   local neg = {}
   local loss = 0
   for k = 1, target:size(1) do
      if target[k] >= 1 then
	 table.insert(pos, input[k])
      else
	 table.insert(neg, input[k])
      end
   end
   for k = 1, #pos do
      for kk = 1, #neg do
         loss = loss + math.exp(neg[kk] - pos[k])
      end
   end
   self.output = loss / #pos / #neg
   return self.output
end

function PWECriterion:updateGradInput(input, target)
   local temp = input.new()
   local lsize = target:sum()
   local nlsize = target:size(1) - lsize
   local mat = torch.FloatTensor(lsize, nlsize)
   input = input:float()
   target = target:float()
   local tmp = input[target:le(0.5)]
   for k = 1, mat:size(1) do
      mat[k] = tmp
   end
   tmp = input[target:ge(0.5)]
   for k = 1, mat:size(2) do
      mat:select(2, k):add(-1, tmp)
   end
   mat = mat:exp() / nlsize / lsize
   self.gradInput = input.new()
   self.gradInput:resizeAs(input)
   self.gradInput[target:ge(0.5)] = -mat:sum(2)
   self.gradInput[target:le(0.5)] = mat:sum(1)
   self.gradInput = self.gradInput:typeAs(temp)
   return self.gradInput
end
