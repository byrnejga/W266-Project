	Cr2q??w@Cr2q??w@!Cr2q??w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Cr2q??w@TT?J?p@1?m????\@ARԙ{H???I??D.8s!@*	{?G?Va@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???_#??!??g>??A@)<?$???1J0??m?<@:Preprocessing2F
Iterator::Model??3??Ŭ?!-i?>?AD@)??B??Ԟ?1??k֥?5@:Preprocessing2U
Iterator::Model::ParallelMapV281$'???!?"!???2@)81$'???1?"!???2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat7?n?e??!Д??x0@)??G7?1@v???i*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?m??4??!r?????@)?m??4??1r?????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip31]????!Ӗ9?L?M@)???VC?~?1!?V?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorY4???r?!??]! 
@)Y4???r?1??]! 
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'jin????!'??#?B@)5)?^?h?1>?R??y@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI`fI?gQ@Q?f?B`>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	TT?J?p@TT?J?p@!TT?J?p@      ??!       "	?m????\@?m????\@!?m????\@*      ??!       2	Rԙ{H???Rԙ{H???!Rԙ{H???:	??D.8s!@??D.8s!@!??D.8s!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q`fI?gQ@y?f?B`>@