	?F??x@?F??x@!?F??x@	??"?Ϸ???"?Ϸ?!??"?Ϸ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?F??x@ܷZ'nPp@1-
?(z?\@A7T??7???I߿yq? @Y?Rb????*	??Q?2f@2U
Iterator::Model::ParallelMapV2~T?~O??!??\{?V6@)~T?~O??1??\{?V6@:Preprocessing2F
Iterator::Model~T?~O???!?t???E@)?*2: 	??1)LBz??4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?|?r????!2??.3?6@)˻?????1IN??Q3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatein??K??!? ???:@)fKVE?ɠ?1??ֽv2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??????!ͬb?_? @)??????1ͬb?_? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipZEh?ɹ?!n?0?\L@)? ?X4?}?1O?1 I@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorW	?3?z?!B???j@)W	?3?z?1B???j@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?i?L???!?'?l?=@)?ُ?au?1??x???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??"?Ϸ?Ib??=?uQ@Q?????>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ܷZ'nPp@ܷZ'nPp@!ܷZ'nPp@      ??!       "	-
?(z?\@-
?(z?\@!-
?(z?\@*      ??!       2	7T??7???7T??7???!7T??7???:	߿yq? @߿yq? @!߿yq? @B      ??!       J	?Rb?????Rb????!?Rb????R      ??!       Z	?Rb?????Rb????!?Rb????b      ??!       JGPUY??"?Ϸ?b qb??=?uQ@y?????>@