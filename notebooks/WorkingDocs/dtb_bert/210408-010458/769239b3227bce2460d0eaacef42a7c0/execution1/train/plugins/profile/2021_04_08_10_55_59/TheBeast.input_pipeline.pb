	W?fşw@W?fşw@!W?fşw@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-W?fşw@?kCŸ?o@1v??=?\@AE)!XU/??I?z??"@*	dX9?h\@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??z?ю??!???ƿ?@@)?? ?K??1)???u<@:Preprocessing2U
Iterator::Model::ParallelMapV2??ɍ"k??!??$?H9@)??ɍ"k??1??$?H9@:Preprocessing2F
Iterator::Model??S9?)??!?iBf?E@)?;??????1??_??1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat<k?]h???!)?<??0@)?q?@H??1???Q?+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???mz?!???'l@)???mz?1???'l@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??I???!U????_L@)R(__?r?1??^IB@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?̯? ?l?!Q????@)?̯? ?l?1Q????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??:?Ϥ?!~??Ϸ?A@)n??d?1?ߐ???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIź??
\Q@Q?a)ԏ>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?kCŸ?o@?kCŸ?o@!?kCŸ?o@      ??!       "	v??=?\@v??=?\@!v??=?\@*      ??!       2	E)!XU/??E)!XU/??!E)!XU/??:	?z??"@?z??"@!?z??"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qź??
\Q@y?a)ԏ>@