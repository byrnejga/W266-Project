	4+ۇ??w@4+ۇ??w@!4+ۇ??w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-4+ۇ??w@???X??o@1??e6?\@A??-????I?ѩ+?#@*	U㥛?e@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???׬?!V?P?@@)?E?~???1P?/?7@:Preprocessing2F
Iterator::Model8i???!$?B@)(?H0?̢?1r8????5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat$d ?.ߢ?!+?@%?5@)_b,?/??1??5?? 2@:Preprocessing2U
Iterator::Model::ParallelMapV2?j???u??!??}?A?.@)?j???u??1??}?A?.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???????!p??|?$@)???????1p??|?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip`?|x? ??!??9pO@)????2??1c????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Q,??z?!m)?z?@)??Q,??z?1m)?z?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptE)!XU??!z.??4(B@)QN????s?1c??lM@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?:?R?hQ@Q>???\>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???X??o@???X??o@!???X??o@      ??!       "	??e6?\@??e6?\@!??e6?\@*      ??!       2	??-??????-????!??-????:	?ѩ+?#@?ѩ+?#@!?ѩ+?#@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?:?R?hQ@y>???\>@