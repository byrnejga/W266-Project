	?}??Di@?}??Di@!?}??Di@	?Z??@????Z??@???!?Z??@???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?}??Di@?5=((?b@1?CQ?O?D@An?+????I??֥F? @Y7U?q7??*	u?V*d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??>????!?(?? vD@)7p??G??1ݘ?8/<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?8?Z???!??t?"%:@)???Oա?1??Nj??5@:Preprocessing2U
Iterator::Model::ParallelMapV2?_????!㷋6?R.@)?_????1㷋6?R.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?/?'??!?q?z)@)?/?'??1?q?z)@:Preprocessing2F
Iterator::ModelyW=`2??!??@\?9@)5?uX??1O??? %@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipl	??g???!??/???R@)˂???:??1b??N?G@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?i?*?~?!????8@)?i?*?~?1????8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?k????!Y3???/E@)???U+c?1?P??E5??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 75.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?Z??@???I????~?S@Q??M?4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?5=((?b@?5=((?b@!?5=((?b@      ??!       "	?CQ?O?D@?CQ?O?D@!?CQ?O?D@*      ??!       2	n?+????n?+????!n?+????:	??֥F? @??֥F? @!??֥F? @B      ??!       J	7U?q7??7U?q7??!7U?q7??R      ??!       Z	7U?q7??7U?q7??!7U?q7??b      ??!       JGPUY?Z??@???b q????~?S@y??M?4@