	%?/???w@%?/???w@!%?/???w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-%?/???w@?zj???o@1??^?2&]@A??? ¯?I؜?g"#@*	???Q?_@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate? ??^???!g?4?>@){??????1&?x??8@:Preprocessing2F
Iterator::ModelLk??^??!_????+E@)?H?,|}??1?rUh?6@:Preprocessing2U
Iterator::Model::ParallelMapV2DP5z5@??!ۋ????3@)DP5z5@??1ۋ????3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatP???(	??!:2"b]3@)????u6??1ү??D/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?=~o??!??????@)?=~o??1??????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?5??Ң??!? VP?L@):̗`}?1?j?|??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	3m??Js?!?Ҷ%(?@)	3m??Js?1?Ҷ%(?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapt??%??!-z/??T@@)?3??k?b?19?-?*F??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIh??	_Q@Q`?U?ۃ>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?zj???o@?zj???o@!?zj???o@      ??!       "	??^?2&]@??^?2&]@!??^?2&]@*      ??!       2	??? ¯???? ¯?!??? ¯?:	؜?g"#@؜?g"#@!؜?g"#@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qh??	_Q@y`?U?ۃ>@