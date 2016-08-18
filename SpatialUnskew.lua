local SpatialUnskew, parent =
   torch.class('nn.SpatialUnskew', 'nn.Module')

function SpatialUnskew:__init(mode)
   parent.__init(self)
   if mode == 'right' then
      self.mode = 1
   else 
      self.mode = -1
   end
end

function SpatialUnskew:updateOutput(input)
   if input:dim() == 3 or input:dim() == 4 then
      output = input.nn.SpatialUnskew_updateOutput(self, input)
   else
      error('input must be 3 or 4-dimensional')
   end
   return output
end

function SpatialUnskew:updateGradInput(input, gradOutput)
   if input:dim() == 3 and gradOutput:dim() == 3 then
      assert(input:size(1) == gradOutput:size(1)
             and input:size(2) == gradOutput:size(2)
             and input:size(3) - input:size(2) + 1 == gradOutput:size(3),
             'input and gradOutput must be compatible in size')
   elseif input:dim() == 4 and gradOutput:dim() == 4 then
      assert(input:size(1) == gradOutput:size(1)
             and input:size(2) == gradOutput:size(2)
             and input:size(3) == gradOutput:size(3)
             and input:size(4) - input:size(3) + 1 == gradOutput:size(4),
             'input and gradOutput must be compatible in size')
   else
      error(
         [[input and gradOutput must be 3 or 4-dimensional
         and have equal number of dimensions]]
         )
   end
   local gradInput = input.nn.SpatialUnskew_updateGradInput(self,input,gradOutput)
   return gradInput
end


