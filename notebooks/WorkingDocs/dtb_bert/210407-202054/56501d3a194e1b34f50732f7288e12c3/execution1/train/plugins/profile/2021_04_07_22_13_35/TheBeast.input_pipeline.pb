	P?}:??h@P?}:??h@!P?}:??h@	????*ݳ?????*ݳ?!????*ݳ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6P?}:??h@???a?b@1?l????D@A7?7M???I ??q@YHO?C????*	^?I3[@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?FtϺF??!ք??jMA@)?-X???1??I??<@:Preprocessing2U
Iterator::Model::ParallelMapV2û\?wb??!\'r֫4@)û\?wb??1\'r֫4@:Preprocessing2F
Iterator::ModelFD1y̤?!??Xp֪B@)??.?5??1?Q?
>1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeataS?Q???!?????5@)\?	????1??(???0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?C4???y?!?C???2@)?C4???y?1?C???2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Nt??!qC??)UO@)g??j+?w?1B(???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorD? ??s?!T@D??@)D? ??s?1T@D??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?aL?{)??!?ֆ?B@)?[[%X\?1%???q??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 75.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????*ݳ?IRw???S@QvG??4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???a?b@???a?b@!???a?b@      ??!       "	?l????D@?l????D@!?l????D@*      ??!       2	7?7M???7?7M???!7?7M???:	 ??q@ ??q@! ??q@B      ??!       J	HO?C????HO?C????!HO?C????R      ??!       Z	HO?C????HO?C????!HO?C????b      ??!       JGPUY????*ݳ?b qRw???S@yvG??4@