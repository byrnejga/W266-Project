	l??gUx@l??gUx@!l??gUx@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-l??gUx@?UסAp@1???7-]@A{?%T??I???4?\ @*	???(\sa@2F
Iterator::Model?2p@KW??!'???1uL@)??<???1j?ϼ??<@:Preprocessing2U
Iterator::Model::ParallelMapV2o~?D???!??/??<@)o~?D???1??/??<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??S?ơ?!uz??8@)??<????1??+ɛ?3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatO???i??!?=*?1)+@)/?e?????1BhE?3y%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???_vO~?!?\%??3@)???_vO~?1?\%??3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????ˮ?!?V +ΊE@)?3??k?r?1?? ??y
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?T???Bp?!U????@)?T???Bp?1U????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?G,???!?tǦ?1:@)???_vO^?1?\%??3??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??plQ@Q?=?O>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?UסAp@?UסAp@!?UסAp@      ??!       "	???7-]@???7-]@!???7-]@*      ??!       2	{?%T??{?%T??!{?%T??:	???4?\ @???4?\ @!???4?\ @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??plQ@y?=?O>@