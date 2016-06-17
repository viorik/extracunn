local SplitTableMultiple, parent = torch.class('nn.SplitTableMultiple', 'nn.Module')

function SplitTableMultiple:__init(dimension, nInputDims, blockSize)
   parent.__init(self)
   self.dimension = dimension
   self.nInputDims = nInputDims
   self.blockSize = blockSize or 1
end

function SplitTableMultiple:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function SplitTableMultiple:updateOutput(input)
   local dimension = self:_getPositiveDimension(input)
   local slices = input:size(dimension)
   assert(math.fmod(slices,self.blockSize)==0,'Dimension must be multiple of blockSize')

   local currentOutput= {}
   for i=1,slices,self.blockSize do
      currentOutput[#currentOutput+1] = input:narrow(dimension,i,self.blockSize):contiguous()
   end
   self.output = currentOutput
   return self.output
end 

function SplitTableMultiple:updateGradInput(input, gradOutput)
   local dimension = self:_getPositiveDimension(input)
   local slices = #gradOutput -- input:size(dimension)
   if self.gradInput then
      self.gradInput:resizeAs(input)

      for i=1,slices do 
         local currentGradInput = gradOutput[i];        
         self.gradInput:narrow(dimension,(i-1)*self.blockSize+1,self.blockSize):copy(currentGradInput)
      end
   end
   return self.gradInput
end

