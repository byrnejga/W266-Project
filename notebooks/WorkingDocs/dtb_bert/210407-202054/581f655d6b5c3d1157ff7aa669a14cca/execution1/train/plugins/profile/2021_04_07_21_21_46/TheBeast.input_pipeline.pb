	^??I??w@^??I??w@!^??I??w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-^??I??w@rp????o@1?Վ??\@A?$????I?,?"?Z @*	R????b@2U
Iterator::Model::ParallelMapV2?????@??!.9@\?f6@)?????@??1.9@\?f6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateXU/??d??!>E??=@)0??!??1??7z??4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??-@??!??n?W?8@)ᛦ????1?s?V?3@:Preprocessing2F
Iterator::Model?+d????!?C5?AC@)]?&?Ҙ?1N*??0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice攀????!???bB @)攀????1???bB @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?e?%⭷?!]???@?N@)?6qr?C??1?6=j@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!?!Ǜ??C5@)ŏ1w-!?1Ǜ??C5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???X?ʧ?!#?J??>@)????9]f?1!?m?)	??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIN?D_Q@Q???˃>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	rp????o@rp????o@!rp????o@      ??!       "	?Վ??\@?Վ??\@!?Վ??\@*      ??!       2	?$?????$????!?$????:	?,?"?Z @?,?"?Z @!?,?"?Z @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qN?D_Q@y???˃>@