	 ????x@ ????x@! ????x@	?F??ږ??F??ږ?!?F??ږ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6 ????x@??^aAp@1???:s`@A,?/o???II?<?+j"@Yq㊋???*	-????e@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatemXSYv??!?2??C@)?j?0
??1f?,Y=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat`?eM,???!?Y7??8@)?.??Ҡ?1ϗK?2@:Preprocessing2F
Iterator::Model?P??9??!uG"???;@)f????l??1?Mqc?/@:Preprocessing2U
Iterator::Model::ParallelMapV2???????!???`$(@)???????1???`$(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?$y??Ñ?!j[??#@)?$y??Ñ?1j[??#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??? ??!"n???R@)1??f???1?????"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Վ?u??!?ӣ<?@)?Վ?u??1?ӣ<?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?V??y??!k?f@h?D@)*??g\8p?1$?Hs?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?F??ږ?I?{?3:?P@Q?oHC?D@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??^aAp@??^aAp@!??^aAp@      ??!       "	???:s`@???:s`@!???:s`@*      ??!       2	,?/o???,?/o???!,?/o???:	I?<?+j"@I?<?+j"@!I?<?+j"@B      ??!       J	q㊋???q㊋???!q㊋???R      ??!       Z	q㊋???q㊋???!q㊋???b      ??!       JGPUY?F??ږ?b q?{?3:?P@y?oHC?D@@