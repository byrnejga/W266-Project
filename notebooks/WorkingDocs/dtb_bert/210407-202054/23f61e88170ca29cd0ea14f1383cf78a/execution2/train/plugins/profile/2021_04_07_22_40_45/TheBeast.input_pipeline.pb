	>?
Y?w@>?
Y?w@!>?
Y?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails->?
Y?w@??,?p@1N|??8?\@A?2?FY??I?H?]"@*	0?$?e^@2F
Iterator::ModelQٰ??(??!????F@)B_z?sѠ?1??)*;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!/?C3??=@)?I/???1????P7@:Preprocessing2U
Iterator::Model::ParallelMapV2?lV}???!?"???72@)?lV}???1?"???72@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?9?w???! ']??1@)??%:?,??1Z???2-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice~?k?,	??!=p;?q?@)~?k?,	??1=p;?q?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)YNB???!\?#?bK@)-[닄?|?1vC?m?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???Q?n?!?g?F??@)???Q?n?1?g?F??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$*T7??!?????Q?@)ŏ1w-!_?1??>?? ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIy?JAcqQ@Q???r:>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??,?p@??,?p@!??,?p@      ??!       "	N|??8?\@N|??8?\@!N|??8?\@*      ??!       2	?2?FY???2?FY??!?2?FY??:	?H?]"@?H?]"@!?H?]"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qy?JAcqQ@y???r:>@