	???Cޭw@???Cޭw@!???Cޭw@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???Cޭw@xԘ??o@1Xq??0]@A??h?????I>x?҆K"@*	cX9?X^@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateo??ܚt??!V5,??M?@)???0????1˒VG??8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I}Yک??!U??q?k<@)Z?h9?C??1?v&=?7@:Preprocessing2U
Iterator::Model::ParallelMapV2?jdWZF??!ee?	k#5@)?jdWZF??1ee?	k#5@:Preprocessing2F
Iterator::Modelg)YNB???!????@@)?ϛ?T??10?{?X?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?7????!'?V/ŗ@)?7????1'?V/ŗ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??????!?.?"??P@)?-:Yj??1??W??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorV?F???x?!?"?Q?@)V?F???x?1?"?Q?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapc?dU????!s??4?@@);?O??nb?1 ??O???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?׫?]WQ@Q??P!??>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	xԘ??o@xԘ??o@!xԘ??o@      ??!       "	Xq??0]@Xq??0]@!Xq??0]@*      ??!       2	??h???????h?????!??h?????:	>x?҆K"@>x?҆K"@!>x?҆K"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?׫?]WQ@y??P!??>@