	+ۇ??x@+ۇ??x@!+ۇ??x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-+ۇ??x@??{Ap@1??O??\@A?}V?)???I?l\?!@*	J+??a@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???ϝ`??!U???V@@)??-@ۢ?1????v?9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat6#??E???!?F??%?9@)N?#~???1E??.,?4@:Preprocessing2F
Iterator::Model???????!?h??%?A@)????~ݙ?1?;.Q??1@:Preprocessing2U
Iterator::Model::ParallelMapV2?3?c?=??!?e1@)?3?c?=??1?e1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?ʉv??!??6???@)?ʉv??1??6???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipaS?Q???!?ˍ=?1P@)?i?*?~?1???o??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorvP??W|?!??6???@)vP??W|?1??6???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapSv?A]???!BTZvs?@@)@j'?;d?1???n????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI0?R?{Q@Q>???>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??{Ap@??{Ap@!??{Ap@      ??!       "	??O??\@??O??\@!??O??\@*      ??!       2	?}V?)????}V?)???!?}V?)???:	?l\?!@?l\?!@!?l\?!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q0?R?{Q@y>???>@