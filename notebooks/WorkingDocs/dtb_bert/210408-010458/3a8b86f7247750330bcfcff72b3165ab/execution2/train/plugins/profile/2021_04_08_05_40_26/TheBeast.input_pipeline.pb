	??8+bx@??8+bx@!??8+bx@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??8+bx@?V??PAp@1m?Yg|?\@A?W???T??I?????!@*	??/ݔ^@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatel?????!?,????@@)?P?R??1?h?f??;@:Preprocessing2F
Iterator::Model???i???!`??? DE@)??.m8,??1?I??J7@:Preprocessing2U
Iterator::Model::ParallelMapV2?ฌ???!??*>3@)?ฌ???1??*>3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ګ???!???D@?2@)h?4?;??1F??'0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????%~?!+?H@)?????%~?1+?H@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?e?????!?YQ<߻L@)?kC?8s?1'??0C!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor9??v??j?!{?U?2A@)9??v??j?1{?U?2A@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?^Cp\ƥ?!&_??*bA@)?N^?U?1hM?J????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIXZ?d??Q@Q???l?=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?V??PAp@?V??PAp@!?V??PAp@      ??!       "	m?Yg|?\@m?Yg|?\@!m?Yg|?\@*      ??!       2	?W???T???W???T??!?W???T??:	?????!@?????!@!?????!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qXZ?d??Q@y???l?=@