	,?j??w@,?j??w@!,?j??w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-,?j??w@y???0p@1?XP?]@AI?Q}??I?????@*	hffffBa@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??????!N??]B@)J?????1?TϽ??=@:Preprocessing2F
Iterator::Model<??X????!V??I?C@)Z)r?#??1)_??6@:Preprocessing2U
Iterator::Model::ParallelMapV2?c?????!?er?n1@)?c?????1?er?n1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatQf?L2r??!3?]8?/@)?ŊLÐ?1^??M5?'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceH?c?C??!W??Yk@)H?c?C??1W??Yk@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?0??f??!??|?EN@)z?}?֤{?1K?;?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorA??ǘ?v?!?_P@)A??ǘ?v?1?_P@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Dׅ??!??Q? ?C@)??E?>q?1?+?}d@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?@??eQ@Q???gh>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	y???0p@y???0p@!y???0p@      ??!       "	?XP?]@?XP?]@!?XP?]@*      ??!       2	I?Q}??I?Q}??!I?Q}??:	?????@?????@!?????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?@??eQ@y???gh>@