	???QO?w@???QO?w@!???QO?w@	?	???,???	???,??!?	???,??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???QO?w@??ңip@1???7?\@Ab?k_@??I?i?쀓!@Y??N????*	cX9?]@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??;???!?$:@@)d??A%??1m?	OFY9@:Preprocessing2F
Iterator::Modelb?o???!ǫ9?F@)?֊6ǹ??1*???8@:Preprocessing2U
Iterator::Model::ParallelMapV2;R}?%??!d>6?M4@);R}?%??1d>6?M4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?h[?:??!?=?Tz+0@)?3??????1?!????*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?=~o??!???? ?@)?=~o??1???? ?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Za?^C??!?8T??YK@)/?$?u?1g%;@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor%]3?f?k?!?g?]?6@)%]3?f?k?1?g?]?6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapgs?69??!m?hA@)?lscz?b?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?	???,??Ik?]1ggQ@QM?s6H>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ңip@??ңip@!??ңip@      ??!       "	???7?\@???7?\@!???7?\@*      ??!       2	b?k_@??b?k_@??!b?k_@??:	?i?쀓!@?i?쀓!@!?i?쀓!@B      ??!       J	??N??????N????!??N????R      ??!       Z	??N??????N????!??N????b      ??!       JGPUY?	???,??b qk?]1ggQ@yM?s6H>@