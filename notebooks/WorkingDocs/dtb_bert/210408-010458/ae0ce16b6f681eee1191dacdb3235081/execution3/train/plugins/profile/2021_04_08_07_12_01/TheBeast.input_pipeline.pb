	?**?x@?**?x@!?**?x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?**?x@?c@??=p@1gҦ??w`@A?}?e?į?IA-Ӗ @*	cX9?h\@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??@?9w??!?G%Oy?@@)5bf??(??1???*?:@:Preprocessing2F
Iterator::Model?0????!?¤4F@)??u?ӝ?1???3?9@:Preprocessing2U
Iterator::Model::ParallelMapV2;:?Fv???!X?r?5?2@);:?Fv???1X?r?5?2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Y?h9Г?!??<??1@)~??$????19??iC?)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicem??!???΅?@)m??1???΅?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??1ZGUs?!??}?L?@)??1ZGUs?1??}?L?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?蜟?8??!?=[??K@)???9]s?11U"?;g@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapkE???&??!}mZuiQA@)?N^?U?1H??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?sa?P@Q?=??w@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?c@??=p@?c@??=p@!?c@??=p@      ??!       "	gҦ??w`@gҦ??w`@!gҦ??w`@*      ??!       2	?}?e?į??}?e?į?!?}?e?į?:	A-Ӗ @A-Ӗ @!A-Ӗ @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?sa?P@y?=??w@@