local MSSECriterion, parent = torch.class('nn.MSSECriterion', 'nn.Criterion')

function MSSECriterion:__init()
   parent.__init(self)
end

function MSSECriterion:updateOutput(input, target)
   return input.nn.MSSECriterion_updateOutput(self, input, target)
end

function MSSECriterion:updateGradInput(input, target)
   return input.nn.MSSECriterion_updateGradInput(self, input, target)
end
