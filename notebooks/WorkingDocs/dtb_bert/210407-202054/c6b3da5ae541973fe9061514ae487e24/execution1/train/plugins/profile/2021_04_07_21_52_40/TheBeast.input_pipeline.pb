	?=δ?x@?=δ?x@!?=δ?x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?=δ?x@c??ޚ?o@1n4??@9`@AN?=??j??I??}?u?"@*	J+??d@2F
Iterator::Model??'*ְ?!8?3H??C@)W??U???1xkPUG6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateǄ?K????!.J?9?H@@)ZI+?????1?&???5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????????!>?b06@)V???5??1???K?V2@:Preprocessing2U
Iterator::Model::ParallelMapV2,?z??m??!??S@?J1@),?z??m??1??S@?J1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?v????!??Rw?&@)?v????1??Rw?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip(?N>=???!?K̷7N@)????%?}?1%C2>@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-C??6z?!d??M6?@)-C??6z?1d??M6?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapςP??Ѭ?!D},???@@)??+ٱa?1?b?t1???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 64.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI=??S??P@Q???X?p@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	c??ޚ?o@c??ޚ?o@!c??ޚ?o@      ??!       "	n4??@9`@n4??@9`@!n4??@9`@*      ??!       2	N?=??j??N?=??j??!N?=??j??:	??}?u?"@??}?u?"@!??}?u?"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q=??S??P@y???X?p@@