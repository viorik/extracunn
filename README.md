#extracunn

This package contains cuda implementations for various layers:
* SpatialConvolutionNoBias: convolutional layer without bias, useful for e.g. gating operations in spatial LSTM cells
* Huber penalty: penalise for non-smoothness
* MSSECriterion: scale-invariant loss layer, useful for e.g. depth estimation. Implementation from [Eigen's paper](http://arxiv.org/pdf/1411.4734v4.pdf). Note that the error must be computer on log ground truth and log prediction.
* InterleaveTable: interleave elements of `n` tables, each one containing `length` elements.

####Installation

Type 'luarocks make' inside the directory to install this package.
