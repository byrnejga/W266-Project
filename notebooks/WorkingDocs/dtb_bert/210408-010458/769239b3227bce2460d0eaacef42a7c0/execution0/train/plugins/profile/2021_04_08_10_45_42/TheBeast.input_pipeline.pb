	!??q4x@!??q4x@!!??q4x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-!??q4x@pA?lLp@1??'*?\@A????kz??I?F??R?@*	@5^?I?d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ګ????!??sw?A@)?$???1,I?̾7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat& ??*Q??!?p?e:@)?ͩd ???16ʭ???5@:Preprocessing2F
Iterator::Model?j?=&R??!=?H??>@)???D??1???]	3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??/ע??!?s?3`R)@)??/ע??1?s?3`R)@:Preprocessing2U
Iterator::Model::ParallelMapV2K???>??!+?o>?'@)K???>??1+?o>?'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?y?Տ??!?????LQ@)?P??C???1a???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???QI}?!???1=#@)???QI}?1???1=#@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???8??!?}?P?B@) ?o_?i?1"??[3??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI????2wQ@Q8?=?4#>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	pA?lLp@pA?lLp@!pA?lLp@      ??!       "	??'*?\@??'*?\@!??'*?\@*      ??!       2	????kz??????kz??!????kz??:	?F??R?@?F??R?@!?F??R?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????2wQ@y8?=?4#>@