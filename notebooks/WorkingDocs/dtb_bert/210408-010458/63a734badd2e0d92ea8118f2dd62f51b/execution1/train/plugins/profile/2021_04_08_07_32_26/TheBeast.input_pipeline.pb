	?4?B?w@?4?B?w@!?4?B?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?4?B?w@?Q{?o@1?????]@Aƅ!Y???I???2"@*	R????`@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??o?4(??!*?B?IC@)ǟ?lXS??1??(?9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????????!0???6@)?????X??1 ???71@:Preprocessing2U
Iterator::Model::ParallelMapV2??-?熖?!;}u?0@)??-?熖?1;}u?0@:Preprocessing2F
Iterator::Model?W??"???!'?{X?>@)??*?]g??1?m?ŝ,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicel?˸???!??\??*@)l?˸???1??\??*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?W)?k??!7}?)EQ@)??X???1?W? ?	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?BB?z?!? ?0!?@)?BB?z?1? ?0!?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapL?Qԙ??!?S}?EZD@)?????g?19????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI&?U ?]Q@Qf?????>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Q{?o@?Q{?o@!?Q{?o@      ??!       "	?????]@?????]@!?????]@*      ??!       2	ƅ!Y???ƅ!Y???!ƅ!Y???:	???2"@???2"@!???2"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q&?U ?]Q@yf?????>@