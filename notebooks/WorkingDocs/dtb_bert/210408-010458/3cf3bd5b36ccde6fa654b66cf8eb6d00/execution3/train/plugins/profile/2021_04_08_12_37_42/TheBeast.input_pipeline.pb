	"ĕsmx@"ĕsmx@!"ĕsmx@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-"ĕsmx@?|AɈp@1????}I]@A6=((E+??I??W9*"@*	?? ?r?e@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??;?%??!????0HB@)X?|[?T??1???j:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?'?????!٣?M??7@)<.?ED1??1????v3@:Preprocessing2F
Iterator::Model? {???!?D?>+e@@)?K??$w??17?2?1?2@:Preprocessing2U
Iterator::Model::ParallelMapV2???B????!f?cHIL,@)???B????1f?cHIL,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceYni5$???!???οL$@)Yni5$???1???οL$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????!?ݦ`j?P@)V?F???1??z?C@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??gy?}?!8?;o)?@)??gy?}?18?;o)?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ƃ-v???!????9C@)??]M??j?1?+??9??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIg[????Q@Qc?vi1?=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?|AɈp@?|AɈp@!?|AɈp@      ??!       "	????}I]@????}I]@!????}I]@*      ??!       2	6=((E+??6=((E+??!6=((E+??:	??W9*"@??W9*"@!??W9*"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qg[????Q@yc?vi1?=@