	??*%x@??*%x@!??*%x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??*%x@C??g??o@1???7?G^@A%w?Df.??I?vLݕ?"@*	X9??v?_@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????l???!-_?ˬ?@)??X??+??1??t,O:@:Preprocessing2F
Iterator::ModelxD??????!???\fmH@)s??h????12?;id9@:Preprocessing2U
Iterator::Model::ParallelMapV2?lw???!???~cv7@)?lw???1???~cv7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat]m???{??!?󵩭Q,@)?&p?n??1???nP'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice1?߄B|?!dJ?:~v@)1?߄B|?1dJ?:~v@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipZ???а??!=???I@)8?*5{?u?1?j??\?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?_?5?!j?!????@)?_?5?!j?1????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap~s??o??!??G?l@@)(?XQ?iX?1Aclg????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIT??l)Q@Q???+LZ?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	C??g??o@C??g??o@!C??g??o@      ??!       "	???7?G^@???7?G^@!???7?G^@*      ??!       2	%w?Df.??%w?Df.??!%w?Df.??:	?vLݕ?"@?vLݕ?"@!?vLݕ?"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qT??l)Q@y???+LZ?@