local InterleaveTable, parent = torch.class('nn.InterleaveTable', 'nn.Module')

function InterleaveTable:__init(length,n)
  parent.__init(self)
  self.n = n 
  self.length = length
  self.output = {}
  self.gradInput = {}
end

function InterleaveTable:updateOutput(input)
  assert(self.n * self.length == #input, "dimension mismatch")
  for k,v in ipairs(self.output) do self.output[k] = nil end
  for i=1,self.length do
    self.output[i] = {}
    for j=1,self.n do
      self.output[i][j] = input[i+(j-1)*self.length]
    end
  end
  return self.output
end

function InterleaveTable:updateGradInput(input, gradOutput)
  for i=1,self.length do
    for j=1,self.n do
      self.gradInput[i+(j-1)*self.length]   = gradOutput[i][j]
    end
  end
  return self.gradInput
end 
