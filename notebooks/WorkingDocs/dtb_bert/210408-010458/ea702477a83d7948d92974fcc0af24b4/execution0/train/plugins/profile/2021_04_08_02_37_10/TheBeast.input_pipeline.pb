	??ZD?w@??ZD?w@!??ZD?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??ZD?w@???f?o@1/?>:u]@A?(&o????I??`8?`"@*	??C?^@2F
Iterator::Model?j?0
??!??]4?E@)????;j??1n?u??	7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???pY??!7?"QOA@)h??52??1?#?6@:Preprocessing2U
Iterator::Model::ParallelMapV2
???%???!XF??/3@)
???%???1XF??/3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?::?Fv??!??[?f1@)[^??6S??1?F??
,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?:?f???!E2qE?#)@)?:?f???1E2qE?#)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipk:!tб?!1?? ?L@)?&S?r?1b9{<?8@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?p>??p?!?4?f?
@)?p>??p?1?4?f?
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?8?@d???!?
Z>LB@)?kC?8c?1G,?䦝??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI????SQ@Q0?1??>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???f?o@???f?o@!???f?o@      ??!       "	/?>:u]@/?>:u]@!/?>:u]@*      ??!       2	?(&o?????(&o????!?(&o????:	??`8?`"@??`8?`"@!??`8?`"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????SQ@y0?1??>@