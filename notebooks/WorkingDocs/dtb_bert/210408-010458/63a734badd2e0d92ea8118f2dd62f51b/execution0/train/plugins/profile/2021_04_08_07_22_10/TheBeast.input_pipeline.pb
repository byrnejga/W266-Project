	?b?= x@?b?= x@!?b?= x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?b?= x@X???[p@1?9?}?	]@A?O??e??I??x? @*	?O??n?`@2U
Iterator::Model::ParallelMapV2??4Ԡ?!a?ȁ??8@)??4Ԡ?1a?ȁ??8@:Preprocessing2F
Iterator::ModelS??iT??!8?6??G@)?L?:???1???7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?c?3?%??!.AS_,?;@)?????a??1	E$z+6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatL6l?ۗ?!B?(??h1@)lZ)r???1??f]?,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?U+~??!??;??@)?U+~??1??;??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?,z????!??D?*J@)???{{?1y<?O@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor~oӟ?Hq?!?rڶ:	@)~oӟ?Hq?1?rڶ:	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?}?[?~??!/?0=F?=@)Ϡ???e?1??ݝq??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???2zQ@Q????5>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	X???[p@X???[p@!X???[p@      ??!       "	?9?}?	]@?9?}?	]@!?9?}?	]@*      ??!       2	?O??e???O??e??!?O??e??:	??x? @??x? @!??x? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???2zQ@y????5>@