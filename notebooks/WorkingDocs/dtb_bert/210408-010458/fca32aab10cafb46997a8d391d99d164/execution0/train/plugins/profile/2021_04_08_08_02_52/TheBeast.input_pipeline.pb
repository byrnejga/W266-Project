	i??֦?w@i??֦?w@!i??֦?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-i??֦?w@?X7?o@1cFx{]@A???|?R??I??R?? @*	>
ףpaa@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?4LkӨ?!dJ?oA@)ɒ9?wգ?1????+?;@:Preprocessing2F
Iterator::Model?'eRC??!???v?%C@)Ԟ?sb??1}????h4@:Preprocessing2U
Iterator::Model::ParallelMapV27??VBw??!?)&??1@)7??VBw??1?)&??1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatA?"r??!XD z?3@)q? ????1U1q??.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice+??????!g???h@)+??????1g???h@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip'L?????!wtu?,?N@)????%?}?1i?R??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor='?o|?y?!???5@)='?o|?y?1???5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???P????!} ?H??B@)???B??b?1??4???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?X !ZQ@QX?~{??>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?X7?o@?X7?o@!?X7?o@      ??!       "	cFx{]@cFx{]@!cFx{]@*      ??!       2	???|?R?????|?R??!???|?R??:	??R?? @??R?? @!??R?? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?X !ZQ@yX?~{??>@