	?-??`x@?-??`x@!?-??`x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?-??`x@4?+?p@1tE)!X]@A]?P????ILK @*	?z?G?b@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatee?9:Z??!?g^?B@)B?"LQ.??1R^8??F;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat1?t?????!?????p8@)tE)!XU??1D?!?,4@:Preprocessing2U
Iterator::Model::ParallelMapV2???H???!1???O /@)???H???11???O /@:Preprocessing2F
Iterator::Model??R??!????
?>@)?????1vD?B??.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceG8-x?W??!?_C??%@)G8-x?W??1?_C??%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipעh[ͺ?!K@?Y?AQ@)???P?v??1?с??3@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?S?<z?!C!?v?@)?S?<z?1C!?v?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?gs???!???C@)???B??b?1??y?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 68.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??i?2?Q@QUX?5?=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	4?+?p@4?+?p@!4?+?p@      ??!       "	tE)!X]@tE)!X]@!tE)!X]@*      ??!       2	]?P????]?P????!]?P????:	LK @LK @!LK @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??i?2?Q@yUX?5?=@