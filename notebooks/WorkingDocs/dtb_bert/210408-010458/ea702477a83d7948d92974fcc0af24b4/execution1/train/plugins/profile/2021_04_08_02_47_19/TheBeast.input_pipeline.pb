	??Q x@??Q x@!??Q x@	?u+?a|??u+?a|?!?u+?a|?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??Q x@???T?#p@1L?uT?\@A*???;??I_??Wf#@Y?R#?3???*	:?O??.e@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????+???!?Φ?'?@@)????W???1?ѽ???7@:Preprocessing2U
Iterator::Model::ParallelMapV2?}??g??!ڴ??7@)?}??g??1ڴ??7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV?1?Ҥ?!?????7@)??n??;??1?{???3@:Preprocessing2F
Iterator::Model?1???A??!??.cB@)?g??s???1^(?Fp)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceW_]?Ő?!??;?T#@)W_]?Ő?1??;?T#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipP??????!|????O@)???4??1???"ֻ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-[닄?|?!?	u??@)-[닄?|?1?	u??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?*4?f??!Y???*?A@)?????g?1??zZ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?u+?a|?I?.?rQ@Qz???13>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???T?#p@???T?#p@!???T?#p@      ??!       "	L?uT?\@L?uT?\@!L?uT?\@*      ??!       2	*???;??*???;??!*???;??:	_??Wf#@_??Wf#@!_??Wf#@B      ??!       J	?R#?3????R#?3???!?R#?3???R      ??!       Z	?R#?3????R#?3???!?R#?3???b      ??!       JGPUY?u+?a|?b q?.?rQ@yz???13>@