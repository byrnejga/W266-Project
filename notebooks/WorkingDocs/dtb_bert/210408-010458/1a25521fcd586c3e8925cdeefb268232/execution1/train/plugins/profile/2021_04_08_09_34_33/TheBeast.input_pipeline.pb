	/?Mx@/?Mx@!/?Mx@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-/?Mx@Gw;S3p@1?6???I]@A??J
,??ID??)H"@*	+??d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?X???F??!??ojD@)j4????1E2??q=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??I`s??!I??I:@)
???????1+?X???4@:Preprocessing2U
Iterator::Model::ParallelMapV27??VBw??![G??.@)7??VBw??1[G??.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/?
ҌE??!??[??&@)/?
ҌE??1??[??&@:Preprocessing2F
Iterator::Modelzq?ҥ?!?Pd???9@)??%:?,??1^???z%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT?T?	g??!???OǍR@)L??1%??1s\F+q@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?6qr?C??!u?=?f@)?6qr?C??1u?=?f@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???'*??!_?EeE@)?K⬈j?1?+???Z??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noItb?qgQ@Q?/v8b>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Gw;S3p@Gw;S3p@!Gw;S3p@      ??!       "	?6???I]@?6???I]@!?6???I]@*      ??!       2	??J
,????J
,??!??J
,??:	D??)H"@D??)H"@!D??)H"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qtb?qgQ@y?/v8b>@