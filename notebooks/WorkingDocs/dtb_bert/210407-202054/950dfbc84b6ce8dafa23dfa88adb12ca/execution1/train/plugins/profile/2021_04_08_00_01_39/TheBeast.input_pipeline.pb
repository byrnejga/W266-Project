	???X3x@???X3x@!???X3x@	?0?{~??0?{~?!?0?{~?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???X3x@?x!up@1???\4?\@A{????I?o??- @YԞ?sb??*	+?!a@2U
Iterator::Model::ParallelMapV2???i2???!ԅ??&2?@)???i2???1ԅ??&2?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatemFA????!(?J?_@@)?!S>U??1A??H;@:Preprocessing2F
Iterator::Model?4f??!?=CwUH@)?Ά?3???1*????w1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?N@aÓ?!P^??+,@)?aod??1&vr???%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice:̗`}?!>h?g?@):̗`}?1>h?g?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?(_?B??!¼???I@)??4???r?1??x?u?
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorj>"?Dr?!??;??	
@)j>"?Dr?1??;??	
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????˧?!????d?@@)?	.V?`Z?1j1KgZ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 68.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?0?{~?I*u??>?Q@QH?u?#?=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?x!up@?x!up@!?x!up@      ??!       "	???\4?\@???\4?\@!???\4?\@*      ??!       2	{????{????!{????:	?o??- @?o??- @!?o??- @B      ??!       J	Ԟ?sb??Ԟ?sb??!Ԟ?sb??R      ??!       Z	Ԟ?sb??Ԟ?sb??!Ԟ?sb??b      ??!       JGPUY?0?{~?b q*u??>?Q@yH?u?#?=@