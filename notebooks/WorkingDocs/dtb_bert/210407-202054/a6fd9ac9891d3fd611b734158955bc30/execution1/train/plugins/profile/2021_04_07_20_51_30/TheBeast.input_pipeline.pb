	???mz?w@???mz?w@!???mz?w@	??Q?q????Q?q??!??Q?q??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???mz?w@#?3???o@1??Q,?&]@Adv?S??I>+N?V#@Y%w?Df.??*	Q???`@2U
Iterator::Model::ParallelMapV2D??k???!F>?(h8@)D??k???1F>?(h8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeath??????!???v?<9@)?N?????1f?????3@:Preprocessing2F
Iterator::Model??
???!4?*;E@)r??>s֗?1"??	2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????:8??!4ѝ?ڑ8@)?iQ???1??.=/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?GĔH???!ۍ???!@)?GĔH???1ۍ???!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?n??S}?!d^?Pv6@)?n??S}?1d^?Pv6@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipp??R????!??u???L@)~6rݔ?z?1?Y?ci@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??4`????!*^)?2;@)?#EdX?k?1?g\??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??Q?q??I@?!TQ@Q?=J?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	#?3???o@#?3???o@!#?3???o@      ??!       "	??Q,?&]@??Q,?&]@!??Q,?&]@*      ??!       2	dv?S??dv?S??!dv?S??:	>+N?V#@>+N?V#@!>+N?V#@B      ??!       J	%w?Df.??%w?Df.??!%w?Df.??R      ??!       Z	%w?Df.??%w?Df.??!%w?Df.??b      ??!       JGPUY??Q?q??b q@?!TQ@y?=J?>@