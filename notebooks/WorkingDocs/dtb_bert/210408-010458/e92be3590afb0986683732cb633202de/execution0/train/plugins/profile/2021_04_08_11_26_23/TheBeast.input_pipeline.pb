	??z?w@??z?w@!??z?w@	 ???P?? ???P??! ???P??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??z?w@p$?`S?o@1?w?~]@A:d?w??I@?	?P"@Y?4-?2??*	?? ?r?d@2F
Iterator::ModelrS??ܱ?!?????D@):?,B???1?ym??7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???BΫ?!;??kH@@)??M?q??1?????35@:Preprocessing2U
Iterator::Model::ParallelMapV2R???<H??!???r?Q2@)R???<H??1???r?Q2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF;?I??!u??)?q2@)???lɪ??1yW??,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?k	??g??!????
?&@)?k	??g??1????
?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipe#?#Ը?!??VAM@)???4??1?O??b@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor5A?} R{?!?-*?O?@)5A?} R{?1?-*?O?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap|H??ߠ??!?? ?YA@)?=\r?)m?1RP??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9 ???P??I???k?`Q@Q?l5-w>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	p$?`S?o@p$?`S?o@!p$?`S?o@      ??!       "	?w?~]@?w?~]@!?w?~]@*      ??!       2	:d?w??:d?w??!:d?w??:	@?	?P"@@?	?P"@!@?	?P"@B      ??!       J	?4-?2???4-?2??!?4-?2??R      ??!       Z	?4-?2???4-?2??!?4-?2??b      ??!       JGPUY ???P??b q???k?`Q@y?l5-w>@