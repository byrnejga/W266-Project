	ƨk???w@ƨk???w@!ƨk???w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ƨk???w@????U?o@1???n?]@Ad<J%<???IN?????!@*	?G?z0`@2U
Iterator::Model::ParallelMapV2????c??!9?1瑻;@)????c??19?1瑻;@:Preprocessing2F
Iterator::Modelc}?E??!?]??7?K@)Z_&??1a+??^;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ީ?{???!???j\?7@)??????1??\Y?2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?L???Ɣ?!???-U/@)?7?ܘ???1???V)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice`???Y~?!=??D??@)`???Y~?1=??D??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zips֧?ŭ?!3?\?rF@)?M???Pt?1?0?2+?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor7T??7?p?!?7\	@)7T??7?p?1?7\	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapR&5?ؠ?!????f9@)?p>??`?1U?/c???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIE??[?RQ@Q?B??ƴ>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????U?o@????U?o@!????U?o@      ??!       "	???n?]@???n?]@!???n?]@*      ??!       2	d<J%<???d<J%<???!d<J%<???:	N?????!@N?????!@!N?????!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qE??[?RQ@y?B??ƴ>@