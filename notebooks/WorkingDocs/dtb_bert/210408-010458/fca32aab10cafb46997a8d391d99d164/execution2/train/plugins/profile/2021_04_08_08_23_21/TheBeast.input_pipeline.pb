	`;??w@`;??w@!`;??w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-`;??w@!?1??0p@1???]@A?ۼqR??IrP?Lۗ!@*	rh??|G\@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatev???!*f=??EA@)?@??1p#C??<@:Preprocessing2F
Iterator::Model?v??????!\(?Ӗ'E@)u???a???1??ˤ?8@:Preprocessing2U
Iterator::Model::ParallelMapV2?<HO?C??!??܈~1@)?<HO?C??1??܈~1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?=~o??!As:Ar?0@) ?4?O??1?r`*,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???~?!?????@)???~?1?????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?g͏????!??<,i?L@)????t?1qn? ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?HP?h?!?z??@)?HP?h?1?z??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6;R}???!6??g?5B@)?R\U?]a?1nA?(????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???qQ@Q?#샵8>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!?1??0p@!?1??0p@!!?1??0p@      ??!       "	???]@???]@!???]@*      ??!       2	?ۼqR???ۼqR??!?ۼqR??:	rP?Lۗ!@rP?Lۗ!@!rP?Lۗ!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???qQ@y?#샵8>@