local SpatialMirrorVertical, parent =
   torch.class('nn.SpatialMirrorVertical', 'nn.Module')

function SpatialMirrorVertical:__init()
   parent.__init(self)
end

function SpatialMirrorVertical:updateOutput(input)
   if input:dim() == 3 or input:dim() == 4 then
      output = input.nn.SpatialMirrorVertical_updateOutput(self,input)
   else
      error('input must be 3 or 4-dimensional')
   end
   return output
end

function SpatialMirrorVertical:updateGradInput(input, gradOutput)
   if input:dim() == 3 and gradOutput:dim() == 3 then
      assert(input:size(1) == gradOutput:size(1)
             and input:size(2) == gradOutput:size(2)
             and input:size(3) == gradOutput:size(3),
             'input and gradOutput must be compatible in size')
   elseif input:dim() == 4 and gradOutput:dim() == 4 then
      assert(input:size(1) == gradOutput:size(1)
             and input:size(2) == gradOutput:size(2)
             and input:size(3) == gradOutput:size(3)
             and input:size(4) == gradOutput:size(4),
             'input and gradOutput must be compatible in size')
   else
      error(
         [[input and gradOutput must be 3 or 4-dimensional
         and have equal number of dimensions]]
         )
   end 
   local gradInput = input.nn.SpatialMirrorVertical_updateGradInput(self, input, gradOutput)
   return gradInput
end


