	l#??f?w@l#??f?w@!l#??f?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-l#??f?w@/n?<p@1??U?\@AxG?j????ITV??D?!@*	&??C?_@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate@?&M????!?/?`!?C@)??b??1????Y:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat{Cr2q??!$???'5@)???1ZG??1?j?pg0@:Preprocessing2U
Iterator::Model::ParallelMapV2?<HO?C??!??d/U>/@)?<HO?C??1??d/U>/@:Preprocessing2F
Iterator::Model???N???!j??<@)???4??1LT???e*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???????!??X_Y*@)???????1??X_Y*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??-</??!??||x?Q@)DQ?O?I??1h?z?2@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorWya?x?! ??@)Wya?x?1 ??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapb?Q+L߫?!lܥ?|E@)V?F?q?1	??T?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI,?[??nQ@QNא?=E>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/n?<p@/n?<p@!/n?<p@      ??!       "	??U?\@??U?\@!??U?\@*      ??!       2	xG?j????xG?j????!xG?j????:	TV??D?!@TV??D?!@!TV??D?!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q,?[??nQ@yNא?=E>@