	??Z?w@??Z?w@!??Z?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??Z?w@???	p@1??\?]@A衶? ??I?r??U"@*	???(\?d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate b???4??!?!?o??B@)?5C?(??1?;z'U:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?b?J!???!????r?7@)??;?%??15"h??03@:Preprocessing2F
Iterator::Modelګ?????!?z?,|@@)?g?K6??1???T?0@:Preprocessing2U
Iterator::Model::ParallelMapV2Lk??^??!cH'?QC0@)Lk??^??1cH'?QC0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceS??????!?A?B?%@)S??????1?A?B?%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?2??3??!?B????P@)????4c??1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor5A?} R{?!?f??;@)5A?} R{?1?f??;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??WXp???!?????NC@)Mۿ?Ҥd?13?ZB???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI-???NcQ@QK<|?r>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???	p@???	p@!???	p@      ??!       "	??\?]@??\?]@!??\?]@*      ??!       2	衶? ??衶? ??!衶? ??:	?r??U"@?r??U"@!?r??U"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q-???NcQ@yK<|?r>@