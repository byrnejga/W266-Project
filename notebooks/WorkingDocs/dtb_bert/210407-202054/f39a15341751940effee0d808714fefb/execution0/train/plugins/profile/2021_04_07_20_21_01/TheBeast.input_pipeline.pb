	?9??}b@?9??}b@!?9??}b@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?9??}b@W???x%%@1H2?w??_@A?
b?k_??IK?46#@*	?~j?t?c@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?Sͬ???!????>@)?Sͬ???1????>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?"?tuǢ?!????6@)H??5\???1?Eȕd?2@:Preprocessing2F
Iterator::Modeli9?Cm??!???6?:@)???j?=??1?ъ?&+@:Preprocessing2U
Iterator::Model::ParallelMapV2???Y???!mA8??*@)???Y???1mA8??*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate <?Bus??!?D?NE@)???o{???1F???	)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip-`?????!?L?@R@)????????1?Y*B5?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?%??:?z?!,????F@)?%??:?z?1,????F@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??)????!?[u?pF@)???P?c?1|?vM??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI=???N+@Q^?--?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	W???x%%@W???x%%@!W???x%%@      ??!       "	H2?w??_@H2?w??_@!H2?w??_@*      ??!       2	?
b?k_???
b?k_??!?
b?k_??:	K?46#@K?46#@!K?46#@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q=???N+@y^?--?U@