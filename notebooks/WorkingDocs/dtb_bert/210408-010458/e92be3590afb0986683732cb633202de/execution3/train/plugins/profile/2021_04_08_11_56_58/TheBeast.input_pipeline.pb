	?? Qp`w@?? Qp`w@!?? Qp`w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?? Qp`w@?Ȳ`bo@1???"??\@A????I+P????#@*	)\??? _@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???V_]??!?!V???@@)?/???t??1??P?X?9@:Preprocessing2F
Iterator::Model	^???!׵ze??D@)'??rJ@??1?֪??6@:Preprocessing2U
Iterator::Model::ParallelMapV2?PMI????!??J??3@)?PMI????1??J??3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat"6X8I???!????2@)?˛õړ?18dF??D/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??5?e???!h}nE??@)??5?e???1h}nE??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip|?(B?v??!)J??wM@){?G?zt?1!h? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!K????	@)????Mbp?1K????	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapXU/??d??!D??X?A@)e?z?Fw`?138I????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??TuNQ@Qୢ+?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Ȳ`bo@?Ȳ`bo@!?Ȳ`bo@      ??!       "	???"??\@???"??\@!???"??\@*      ??!       2	????????!????:	+P????#@+P????#@!+P????#@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??TuNQ@yୢ+?>@