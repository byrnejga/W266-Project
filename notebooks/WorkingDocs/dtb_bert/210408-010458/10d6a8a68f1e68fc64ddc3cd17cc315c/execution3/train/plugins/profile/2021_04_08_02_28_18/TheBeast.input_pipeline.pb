	????w@????w@!????w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-????w@?
?p@1Y5s?]@A????B???IQ???!@*	9??v??_@2F
Iterator::Model8/N|????!Z?"?P?I@)?J?ó??1???1:@:Preprocessing2U
Iterator::Model::ParallelMapV2s?4?B??!2zp?8@)s?4?B??12zp?8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??S?ơ?!Ϯ?H;@)A?C???1??R?qO6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??^EF??!?????.@)??$"????1??0w?e)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice='?o|?y?!?o+??@)='?o|?y?1?o+??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipԙ{H?ޯ?!??k?uH@)?O?I?5s?1ى?w|@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_?Q?k?!?U!<`@)_?Q?k?1?U!<`@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapo??ܚt??!#?:5??=@)t^c???j?1?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?????_Q@Q???>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?
?p@?
?p@!?
?p@      ??!       "	Y5s?]@Y5s?]@!Y5s?]@*      ??!       2	????B???????B???!????B???:	Q???!@Q???!@!Q???!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?????_Q@y???>@