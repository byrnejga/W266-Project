	??.??w@??.??w@!??.??w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??.??w@?m???o@1@4???]@A?X"????I??=??7"@*	???Q?d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate}?|??!<??k??D@)????	???1?I??~ <@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??;?2??!\?T_x?7@)??K⬠?1?P?-?3@:Preprocessing2F
Iterator::Model??k????!q???<@)??9?٘?1?=$?,-@:Preprocessing2U
Iterator::Model::ParallelMapV2??????!?t )?*@)??????1?t )?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice VG?t??!?  ??)@) VG?t??1?  ??)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???߽???!c?L??Q@){?Fw;??1?Ճ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??w?'-|?!?]?Ɯ?@)??w?'-|?1?]?Ɯ?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???2#??!??tdVJE@)????d?1????l??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?a?-*KQ@Qxy?IW?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?m???o@?m???o@!?m???o@      ??!       "	@4???]@@4???]@!@4???]@*      ??!       2	?X"?????X"????!?X"????:	??=??7"@??=??7"@!??=??7"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?a?-*KQ@yxy?IW?>@