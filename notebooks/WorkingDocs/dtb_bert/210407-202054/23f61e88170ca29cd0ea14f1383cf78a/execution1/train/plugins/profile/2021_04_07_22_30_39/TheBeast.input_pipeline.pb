	?A|`Gbx@?A|`Gbx@!?A|`Gbx@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?A|`Gbx@?Nw???o@1??`8?o_@A???^??I???啫"@*	??Q??a@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate*A*Ŏ??!?5
!fK?@)JC?B???1?H?{`8@:Preprocessing2U
Iterator::Model::ParallelMapV2??{?_???!4"??%7@)??{?_???14"??%7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?G??'???!h0D?qK:@)5(??ȟ?1+??J6@:Preprocessing2F
Iterator::Model???8a¨?!???t?,A@).?&??1j??(g&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?H?s
???!??멫@)?H?s
???1??멫@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?sD?K???!)?ŬiP@)????16Y?R?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???|~x?!??????@)???|~x?1??????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+n?b~n??!??ZRv?@@) ????m?1?<Y4?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??U4?P@Q ?U?@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Nw???o@?Nw???o@!?Nw???o@      ??!       "	??`8?o_@??`8?o_@!??`8?o_@*      ??!       2	???^?????^??!???^??:	???啫"@???啫"@!???啫"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??U4?P@y ?U?@@