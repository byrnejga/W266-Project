	:$?P2x@:$?P2x@!:$?P2x@	?d????~??d????~?!?d????~?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6:$?P2x@:τ?oo@1yX?5?_@A?̔????I?tYLl?!@Y??b? ̝?*	??Mb\@2F
Iterator::Model?hV?y??!?A??G@)|?E{????1?ϧ?̏:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate8؛????!??l?_?>@)f?y??̛?1?h??(8@:Preprocessing2U
Iterator::Model::ParallelMapV2	?f?ba??!]??7?/5@)	?f?ba??1]??7?/5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat)x
?Rϒ?!m????X0@)?Ēr?9??17?eD*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice)????h}?!DK?a??@))????h}?1DK?a??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??_#I??!?>??, J@)???B??r?1,@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????m?!??W?	@)?????m?1??W?	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap0,?-X??!?mB???@)?~j?t?X?1?!W\[??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?d????~?I????P@Q=?jN?)@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	:τ?oo@:τ?oo@!:τ?oo@      ??!       "	yX?5?_@yX?5?_@!yX?5?_@*      ??!       2	?̔?????̔????!?̔????:	?tYLl?!@?tYLl?!@!?tYLl?!@B      ??!       J	??b? ̝???b? ̝?!??b? ̝?R      ??!       Z	??b? ̝???b? ̝?!??b? ̝?b      ??!       JGPUY?d????~?b q????P@y=?jN?)@@