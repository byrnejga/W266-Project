	3Q??m?`@3Q??m?`@!3Q??m?`@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-3Q??m?`@??x?Z?@1y?Տ?\@A ?????I,?F<ٽ"@*	+??vc@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate:Yj??h??!?@`زC@)q?5鶤?1q,?t?9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ю~7??!a??u??:@)??.?.??1\1??;?5@:Preprocessing2F
Iterator::Model:????!????CD;@)?S9?)9??1xP?	!-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?sb?c??!=!x?*@)?sb?c??1=!x?*@:Preprocessing2U
Iterator::Model::ParallelMapV2C ?8?@??!$3_~g)@)C ?8?@??1$3_~g)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipf`X???!?O??.R@)?????ڀ?10??-$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??V?c#??!@?_>@)??V?c#??1@?_>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap|ds?<G??!#3H?jD@)???מYb?1Wi{=??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI????T?)@Q?"?f??U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??x?Z?@??x?Z?@!??x?Z?@      ??!       "	y?Տ?\@y?Տ?\@!y?Տ?\@*      ??!       2	 ????? ?????! ?????:	,?F<ٽ"@,?F<ٽ"@!,?F<ٽ"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????T?)@y?"?f??U@