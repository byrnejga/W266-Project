	?x}j@?x}j@!?x}j@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?x}j@?mU?c@1+?w?7?D@A???!o??I?????U@*	?~j?t?g@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ؗl<ض?!???=G@)???GS??1???3?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??l#???!??????4@)???8?j??1?Q?D?1@:Preprocessing2F
Iterator::ModelL???j???!?????-:@)???̚?16?P?C+@:Preprocessing2U
Iterator::Model::ParallelMapV2???lɪ??!?9?YM)@)???lɪ??1?9?YM)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?f????!?????u&@)?f????1?????u&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Fx$??!?????tR@)'?ei????1?U)??:@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????y?!?^??A
@)??????y?1?^??A
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???L0???!d? ?H@)???|~h?1tz8C;???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 76.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIO\?c?T@QĎ?q??3@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?mU?c@?mU?c@!?mU?c@      ??!       "	+?w?7?D@+?w?7?D@!+?w?7?D@*      ??!       2	???!o?????!o??!???!o??:	?????U@?????U@!?????U@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qO\?c?T@yĎ?q??3@