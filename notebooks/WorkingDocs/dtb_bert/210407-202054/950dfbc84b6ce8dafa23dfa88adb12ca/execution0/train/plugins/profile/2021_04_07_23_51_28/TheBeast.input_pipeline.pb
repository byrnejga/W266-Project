	"??T2`x@"??T2`x@!"??T2`x@	SN??	My?SN??	My?!SN??	My?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6"??T2`x@9}=_?o@12?????_@A??????I????}!@Y???O???*	?&1??`@2F
Iterator::ModelI?<?+J??!C?'5?\I@)0??&???1?r1?j?;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatem ]lZ)??!z??6?=@)????j؟?1?o˸h[7@:Preprocessing2U
Iterator::Model::ParallelMapV2??'?8??!????6@)??'?8??1????6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM?J???!?6U0?]-@)?????j??1=C??
'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice+???????!8???7?@)+???????18???7?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip? [??˰?!?Q??:?H@)YLl>?u?1(ya?%?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??E?>q?!R?W??K	@)??E?>q?1R?W??K	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????4???!????O?>@)A??h:;Y?1+??F????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9SN??	My?Ix??e??P@Qm??,d@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	9}=_?o@9}=_?o@!9}=_?o@      ??!       "	2?????_@2?????_@!2?????_@*      ??!       2	????????????!??????:	????}!@????}!@!????}!@B      ??!       J	???O??????O???!???O???R      ??!       Z	???O??????O???!???O???b      ??!       JGPUYSN??	My?b qx??e??P@ym??,d@@