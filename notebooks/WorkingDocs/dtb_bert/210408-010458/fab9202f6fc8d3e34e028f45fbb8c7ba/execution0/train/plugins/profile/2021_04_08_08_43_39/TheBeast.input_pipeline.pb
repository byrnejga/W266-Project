	???ҍ=w@???ҍ=w@!???ҍ=w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???ҍ=w@?N?jo@1?/????\@A??9̗??I`?;?!@*	?G?zRd@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??66;R??!c??'?A@)kE???&??1?'O?68@:Preprocessing2F
Iterator::Model?/fKVE??!YΡ?i?C@)?J?ó??1?6??Sr4@:Preprocessing2U
Iterator::Model::ParallelMapV2?)ʥ???!fe??2@)?)ʥ???1fe??2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??h>???!1???s?2@)&5?؀??1(>*vp-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?\7??V??!-???t&@)?\7??V??1-???t&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?7ӅX??!?1^E?sN@);?O??n??1?k??%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????y?!?????@)??????y?1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA*Ŏơ??!e֣?fB@)?h㈵?d?187??12??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?~?>AQ@Q??g?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?N?jo@?N?jo@!?N?jo@      ??!       "	?/????\@?/????\@!?/????\@*      ??!       2	??9̗????9̗??!??9̗??:	`?;?!@`?;?!@!`?;?!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?~?>AQ@y??g?>@