	?	??0?w@?	??0?w@!?	??0?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?	??0?w@??K??o@1???G]@A??)x
??I?\??! @*	???S??c@2F
Iterator::Model?C?H????!?o??jG@)ޯ|?y??1?s???P8@:Preprocessing2U
Iterator::Model::ParallelMapV2g׽?	??!?k ^!?6@)g׽?	??1?k ^!?6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???x???!??$???=@)g)YNB??1FB?b?5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????%:??!B???7?0@)?gyܝ??10!D0?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?w?~?~??!B?????@)?w?~?~??1B?????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip!??nJ??!0???J@)?T???B??1f?pM@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Z	?%qv?!?y???@)?Z	?%qv?1?y???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapƉ?v???!??0z?@)?M???Pd?1e-?@?]??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??Y?VTQ@Q?y?j??>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??K??o@??K??o@!??K??o@      ??!       "	???G]@???G]@!???G]@*      ??!       2	??)x
????)x
??!??)x
??:	?\??! @?\??! @!?\??! @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??Y?VTQ@y?y?j??>@