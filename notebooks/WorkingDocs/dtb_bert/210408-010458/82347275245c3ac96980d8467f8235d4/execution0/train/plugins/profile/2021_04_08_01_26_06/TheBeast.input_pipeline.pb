	*S?A?w@*S?A?w@!*S?A?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-*S?A?w@O????o@1?il?]@A?(&o????I? ???3!@*	^?IsY@2F
Iterator::Model?w?-;ħ?!u?????F@)(?r?w??1???fTn8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????Ν?!?S>p[?<@)r7?֊6??1????D6@:Preprocessing2U
Iterator::Model::ParallelMapV2??????!f??b)+5@)??????1f??b)+5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??B=}??!??????3@)?|?b?:??1????p#/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?	.V?`z?!H??%N@)?	.V?`z?1H??%N@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipj??U?Z??!?lA3K@)?ds?1?@?;|;@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorC?8
q?!y?Y?X@)C?8
q?1y?Y?X@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??'Hlw??!7B???/>@)????ŊZ?1???vbv??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI)?N?MQ@Q[א?J?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	O????o@O????o@!O????o@      ??!       "	?il?]@?il?]@!?il?]@*      ??!       2	?(&o?????(&o????!?(&o????:	? ???3!@? ???3!@!? ???3!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q)?N?MQ@y[א?J?>@