	??9}?x@??9}?x@!??9}?x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??9}?x@\X7?],p@1o???g_@A+?3???IR?r?!@*?A`???g@)       =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateA?º???!?T?C??@)?Df.py??1'?dG?5@:Preprocessing2F
Iterator::Model/???uR??!/?Y??C@)?bd?ˣ?1??&s?R4@:Preprocessing2U
Iterator::Model::ParallelMapV2????٢?!lጟ/[3@)????٢?1lጟ/[3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?y??Q??!?Q?c?5@)-]?6???1xG?\2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceѐ?(????!???K?%@)ѐ?(????1???K?%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipԁ??V_??!?N?v?(N@)kQL? ??1{???J?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???{{?!vP?8@)???{{?1vP?8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?h?^`V??!??[???@@)?@??_?k?1??d?hn??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??#D\Q@Q?p????@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	\X7?],p@\X7?],p@!\X7?],p@      ??!       "	o???g_@o???g_@!o???g_@*      ??!       2	+?3???+?3???!+?3???:	R?r?!@R?r?!@!R?r?!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??#D\Q@y?p????@