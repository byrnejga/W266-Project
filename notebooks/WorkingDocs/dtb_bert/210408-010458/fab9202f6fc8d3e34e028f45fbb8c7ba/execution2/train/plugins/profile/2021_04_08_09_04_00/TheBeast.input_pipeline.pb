	v?1<??w@v?1<??w@!v?1<??w@	?@T??ݷ??@T??ݷ?!?@T??ݷ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6v?1<??w@?Q?Gp@1??`???\@AZH?????I?d73?1"@Y?c"????*	????M\@2F
Iterator::Model???o^??!???6?H@)??e???1\>???<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateV?y?՟?!?#%!??;@)???JY???1????WT5@:Preprocessing2U
Iterator::Model::ParallelMapV2??$?z??!?=??k4@)??$?z??1?=??k4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???o{???!?Ŋ?e?1@)H¾?D???1?q??i+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?k??=}?!?-?un@)?k??=}?1?-?un@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?ds?!+3??z?@)?ds?1+3??z?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipbI????!?TU:?SI@)"??u??q?1\?guԢ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapI?5C???!??rc??<@)?c#??W?13~?$????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?@T??ݷ?I?H#?kQ@Q"???9>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Q?Gp@?Q?Gp@!?Q?Gp@      ??!       "	??`???\@??`???\@!??`???\@*      ??!       2	ZH?????ZH?????!ZH?????:	?d73?1"@?d73?1"@!?d73?1"@B      ??!       J	?c"?????c"????!?c"????R      ??!       Z	?c"?????c"????!?c"????b      ??!       JGPUY?@T??ݷ?b q?H#?kQ@y"???9>@