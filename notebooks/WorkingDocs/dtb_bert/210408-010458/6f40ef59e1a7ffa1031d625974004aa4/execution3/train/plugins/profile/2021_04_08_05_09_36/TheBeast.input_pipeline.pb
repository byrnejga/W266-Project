	l@???'x@l@???'x@!l@???'x@	?ا?N???ا?N??!?ا?N??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6l@???'x@F????cp@1??D??\@AR?GT???I?Ǻ??!@Y???[??*	z?G??c@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateuʣaQ??!?6??р?@)ǂ L???1ނPj?5@:Preprocessing2F
Iterator::Model?????M??!?w?.??B@)?L???$??1G?	E.4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ɐc??!t?g?1 8@)???AB???1?f??3@:Preprocessing2U
Iterator::Model::ParallelMapV2#???R??!?R??1@)#???R??1?R??1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???Q???!?g?|?#@)???Q???1?g?|?#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?M????!?9?g%O@)??K?[??1? ??X?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?>tA}?|?!y???1?@)?>tA}?|?1y???1?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\ qW???!?l??#?@@)s.?Ue?e?1i)??\7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?ا?N??IGǸ? ?Q@Q??Mu?=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	F????cp@F????cp@!F????cp@      ??!       "	??D??\@??D??\@!??D??\@*      ??!       2	R?GT???R?GT???!R?GT???:	?Ǻ??!@?Ǻ??!@!?Ǻ??!@B      ??!       J	???[?????[??!???[??R      ??!       Z	???[?????[??!???[??b      ??!       JGPUY?ا?N??b qGǸ? ?Q@y??Mu?=@