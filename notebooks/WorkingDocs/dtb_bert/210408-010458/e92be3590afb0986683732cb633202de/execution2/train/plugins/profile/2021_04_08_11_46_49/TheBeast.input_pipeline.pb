	?Ѫ???w@?Ѫ???w@!?Ѫ???w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?Ѫ???w@?t???p@1p???? ]@Aa?4?ͯ?I?/???!@*	??Mb_@2F
Iterator::Model??$y???!?????H@)??i????1~dk2~?;@:Preprocessing2U
Iterator::Model::ParallelMapV2?.????!?$XU?6@)?.????1?$XU?6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??m????!P?????;@)?/h!???1∥?5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatϠ?????!B??C??0@)??HV??1Ţy?Q9+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice+??6?~?!?mh?>@)+??6?~?1?mh?>@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip(}!?????!`;<?I@)<?.9?t?1?*a)o@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??b???p?!??>??o
@)??b???p?1??>??o
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?͌~4???!φ??j9=@)6Y???]?1?'?O?i??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI$"???bQ@Qrw???t>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?t???p@?t???p@!?t???p@      ??!       "	p???? ]@p???? ]@!p???? ]@*      ??!       2	a?4?ͯ?a?4?ͯ?!a?4?ͯ?:	?/???!@?/???!@!?/???!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q$"???bQ@yrw???t>@