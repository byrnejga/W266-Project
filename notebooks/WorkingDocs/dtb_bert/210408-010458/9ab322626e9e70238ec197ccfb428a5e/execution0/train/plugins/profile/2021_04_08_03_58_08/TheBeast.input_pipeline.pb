	?	.Vԃw@?	.Vԃw@!?	.Vԃw@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?	.Vԃw@?[???`o@1?$?)?]@A?Ӟ?sb??I'?o|?!"@*	X9??v^]@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?`ũ?¤?!???'BA@)???5[??1??Z?)?<@:Preprocessing2F
Iterator::Model??jGq???!u?G?tF@)?fHū??1L?3t?7@:Preprocessing2U
Iterator::Model::ParallelMapV2???yq??!???uQ4@)???yq??1???uQ4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatw?Nyt??!
X??3,0@)e????c??17VC)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?]K?={?!y?ٖ??@)?]K?={?1y?ٖ??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip? [??˰?!?!?\??K@)'??@js?1?/_?#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorC?8
q?!1???ET@)C?8
q?11???ET@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??n??o??!?6???A@)8?*5{?U?1??eg???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???T8IQ@Q#????>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?[???`o@?[???`o@!?[???`o@      ??!       "	?$?)?]@?$?)?]@!?$?)?]@*      ??!       2	?Ӟ?sb???Ӟ?sb??!?Ӟ?sb??:	'?o|?!"@'?o|?!"@!'?o|?!"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???T8IQ@y#????>@