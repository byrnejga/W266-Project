	 zR&5 i@ zR&5 i@! zR&5 i@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails- zR&5 i@??U?? c@10??mP{D@AOϻ??0??I?KK?@*	?z?GI_@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate? #????!P???rR?@)?혺+???1?Y?<?:@:Preprocessing2F
Iterator::Model??jGq???!OW?!?D@)Ԟ?sb??1??M?H?6@:Preprocessing2U
Iterator::Model::ParallelMapV2eRC???!?!?2??2@)eRC???1?!?2??2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatHū?m??!?????4@)??UG?t??1??	ۅ1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??Q,??z?!?9????@)??Q,??z?1?9????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???H²?!??w??FM@)a??+ey?1??A?'?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?x#??o?!?h?]??@)?x#??o?1?h?]??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??L!??!h?$h?|@@)?????`?1???w??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 76.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIv??.??S@Q*?TDa4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??U?? c@??U?? c@!??U?? c@      ??!       "	0??mP{D@0??mP{D@!0??mP{D@*      ??!       2	Oϻ??0??Oϻ??0??!Oϻ??0??:	?KK?@?KK?@!?KK?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qv??.??S@y*?TDa4@