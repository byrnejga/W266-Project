	?T??#x@?T??#x@!?T??#x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?T??#x@?:?Mp@1????+]@Aӡ??n,??IF{??_!@*	????Cb@2F
Iterator::ModelD?.l?V??!摤?,?H@)gs?69??1???;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate ??X??!j?9??1;@)??1????1Nm?Y6@:Preprocessing2U
Iterator::Model::ParallelMapV2 ??Udt??!???_?5@) ??Udt??1???_?5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???6p??!?x?e1@)?E?????1Un?s?p+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?kЗ??|?!?????`@)?kЗ??|?1?????`@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?&?|???!n[S?|I@)h??n?|?1?{?zZ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??f??u?!I???Qf@)??f??u?1I???Qf@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapC??À??!B??Ƚ<@)???B??b?1}?N? ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?????rQ@Q%?y5>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?:?Mp@?:?Mp@!?:?Mp@      ??!       "	????+]@????+]@!????+]@*      ??!       2	ӡ??n,??ӡ??n,??!ӡ??n,??:	F{??_!@F{??_!@!F{??_!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?????rQ@y%?y5>@