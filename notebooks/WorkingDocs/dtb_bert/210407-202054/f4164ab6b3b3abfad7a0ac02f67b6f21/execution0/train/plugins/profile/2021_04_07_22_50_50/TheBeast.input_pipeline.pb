	{Cr<x@{Cr<x@!{Cr<x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-{Cr<x@W\??o@1????}?_@AV?y?կ?Iۈ'???@*	)\????`@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??аu??!O???^.?@)??zĠ?1J?o6?]8@:Preprocessing2F
Iterator::Model?-?熦??!??d?"?D@)k?	?i???1???5@:Preprocessing2U
Iterator::Model::ParallelMapV2#???R??!Z1?Y/?4@)#???R??1Z1?Y/?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?:???!w??5@)?kBZcЙ?19?? ??2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?lscz?!?Ϲ?B@)?lscz?1?Ϲ?B@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?$xC??!f?k?.M@)?????gv?10	;?vG@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorC?8
q?!??"???@)C?8
q?1??"???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????S??!????8@@)???[?[?1???N?9??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI:)????P@Q???Ff@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	W\??o@W\??o@!W\??o@      ??!       "	????}?_@????}?_@!????}?_@*      ??!       2	V?y?կ?V?y?կ?!V?y?կ?:	ۈ'???@ۈ'???@!ۈ'???@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q:)????P@y???Ff@@