	??a?'?L@??a?'?L@!??a?'?L@	??? ?????? ???!??? ???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??a?'?L@Ժj?U@14GV~VE@A?y??Q???Ist?? @Yscz???*	?MbX?\@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??0Xr??!~?q?DB@)??m????1	Hza?>@:Preprocessing2U
Iterator::Model::ParallelMapV2V?pA???!e>??t8@)V?pA???1e>??t8@:Preprocessing2F
Iterator::Model?R???ҧ?!Z?o?JD@)F?W?????1???U? 0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?j?????!??#1@)7T??7???1?y???T,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????%~?!г?֭@)?????%~?1г?֭@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipw?Df.p??!???=?M@)???խ?s?1?=Z?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor겘?|\k?!?E?xN@)겘?|\k?1?E?xN@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???J#f??!?"??`C@)~t??gy^?1?lJ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?14.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??? ???I`?<?*?9@Q??P?׎R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ժj?U@Ժj?U@!Ժj?U@      ??!       "	4GV~VE@4GV~VE@!4GV~VE@*      ??!       2	?y??Q????y??Q???!?y??Q???:	st?? @st?? @!st?? @B      ??!       J	scz???scz???!scz???R      ??!       Z	scz???scz???!scz???b      ??!       JGPUY??? ???b q`?<?*?9@y??P?׎R@