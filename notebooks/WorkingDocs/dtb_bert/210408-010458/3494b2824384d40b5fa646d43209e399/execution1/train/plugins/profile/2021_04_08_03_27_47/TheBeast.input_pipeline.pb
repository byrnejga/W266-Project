	{??{Ufx@{??{Ufx@!{??{Ufx@	?V???????V??????!?V??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6{??{Ufx@}?Ж?zo@1?,???>`@A?c\qqT??Is߉Y? @Y???H.??*	?Q???]@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?<e5]O??!`?q?#?@@)??=??W??1?4ZMx?:@:Preprocessing2U
Iterator::Model::ParallelMapV2^??jGq??!??<n??8@)^??jGq??1??<n??8@:Preprocessing2F
Iterator::Model/PR`L??!|3??ÞD@)??U?&??1O?-m0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?F???!??,bc2@)???s??1??J+.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?-:Yj??!?X$&>?@)?-:Yj??1?X$&>?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??????!??J?<aM@);?*??y?14?c?(?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??-?lp?!7<\??
@)??-?lp?17<\??
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?V%?}???!??%??A@)n??d?1??c?#\ @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 64.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?V??????Ib??ڪP@Qef??ޤ@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	}?Ж?zo@}?Ж?zo@!}?Ж?zo@      ??!       "	?,???>`@?,???>`@!?,???>`@*      ??!       2	?c\qqT???c\qqT??!?c\qqT??:	s߉Y? @s߉Y? @!s߉Y? @B      ??!       J	???H.?????H.??!???H.??R      ??!       Z	???H.?????H.??!???H.??b      ??!       JGPUY?V??????b qb??ڪP@yef??ޤ@@