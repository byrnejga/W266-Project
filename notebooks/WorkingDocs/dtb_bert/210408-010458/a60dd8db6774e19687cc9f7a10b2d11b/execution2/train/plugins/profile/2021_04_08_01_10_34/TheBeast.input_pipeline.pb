	??ډ??h@??ډ??h@!??ډ??h@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??ډ??h@?7????b@1?HV?D@A?f׽???IC?i?q
 @*	???S??f@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateiq?0'h??!???A??D@)???а??1S<9v?&?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???Or???!??䟃7@)˞6????1?['??#3@:Preprocessing2F
Iterator::Model#?tu?b??!??9J?Q=@)??|?r٠?1B?8;?	2@:Preprocessing2U
Iterator::Model::ParallelMapV2\?O???!A?|?&@)\?O???1A?|?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice)>>!;o??!?E@r?$@))>>!;o??1?E@r?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip ?yrM???!??q퓫Q@)??։ˁ?1?u?? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?? ?X4}?!e??=D@)?? ?X4}?1e??=D@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?I?p??!qB?BoE@)a2U0*?c?1?\!x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 75.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?l%$?S@Q??Mjo?4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?7????b@?7????b@!?7????b@      ??!       "	?HV?D@?HV?D@!?HV?D@*      ??!       2	?f׽????f׽???!?f׽???:	C?i?q
 @C?i?q
 @!C?i?q
 @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?l%$?S@y??Mjo?4@