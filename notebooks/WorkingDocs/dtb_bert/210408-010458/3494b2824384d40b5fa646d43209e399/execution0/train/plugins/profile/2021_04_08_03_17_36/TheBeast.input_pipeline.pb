	??EZ?w@??EZ?w@!??EZ?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??EZ?w@?m???o@1oI???\@AV?y?կ?ItD?K??!@*	???Mb?\@2F
Iterator::Model??l ]??!S?M???G@)???%:˜?1???,?J8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateI?V????! P??=@)?f?|?|??1??{N8@:Preprocessing2U
Iterator::Model::ParallelMapV2?gx????!??	??7@)?gx????1??	??7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?=?????!E!Ms?0@)&???J??1Xڂ?%}+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicet^c???z?!??`c?@)t^c???z?1??`c?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??h>???!?;?d;J@)?ُ?au?1???	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor,???cZk?!̠]?|@),???cZk?1̠]?|@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapw??`??!?RR??@)?E|'f?X?1?'?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??'?YQ@Q?a???>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?m???o@?m???o@!?m???o@      ??!       "	oI???\@oI???\@!oI???\@*      ??!       2	V?y?կ?V?y?կ?!V?y?կ?:	tD?K??!@tD?K??!@!tD?K??!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??'?YQ@y?a???>@