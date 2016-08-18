#extracunn

This package contains cuda implementations for various layers:
* SpatialConvolutionNoBias: convolutional layer without bias, useful for e.g. gating operations in spatial LSTM cells.
* Huber penalty: penalise for non-smoothness.
* MSSECriterion: scale-invariant loss layer, useful for e.g. depth estimation. Implementation from [Eigen's paper](http://arxiv.org/pdf/1411.4734v4.pdf). Note that the error is computed between log ground truth and log prediction.
* InterleaveTable: interleave elements of `n` tables, each one containing `length` elements.
* SplitTableMultiple: split a tensor into tensors of `k` elements each; for `k=1`, this is the same as SplitTable.
* SpatialSkew: skew input feature maps to the right; each row is shifted by one position wrt previous row.
* SpatialUnskew: undo the effect of SpatialSkew.
* SpatialMirrorHorizontal: mirror input tensor along central vertical axis.
* SpatialMirrorVertical: mirror input tensor along central horizontal axis.
* SpatialMirrorDiagonal: SpatialMirrorHorizontal + SpatialMirrorVertical done in one step for efficiency. 

####Installation

Type 'luarocks make' inside the directory to install this package.
