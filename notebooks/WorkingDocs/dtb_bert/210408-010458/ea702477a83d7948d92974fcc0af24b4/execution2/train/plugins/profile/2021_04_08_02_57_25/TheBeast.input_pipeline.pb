	.???w@.???w@!.???w@	??(?3$????(?3$??!??(?3$??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6.???w@?lY?.p@1?=?U?\@A?)??Y???I&??s|? @Y̛õ?Þ?*	p=
ף\c@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateR???0???!??AW?C@)T??b???1N"Cl?K>@:Preprocessing2F
Iterator::Model??ᱟŪ?!D????@@)?/?1"Q??1?	q??4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?}??ŉ??!??j5@?3@)k?C4????1??BH?.@:Preprocessing2U
Iterator::Model::ParallelMapV2b?? ????!???o?]*@)b?? ????1???o?]*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?;?D??!ǡ-z#@)?;?D??1ǡ-z#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??_cD??!?]2???P@)?7????1 JW??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;???.|?!??$?o?@);???.|?1??$?o?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??jQ??!??ĳ?D@)y=??`?1??PGn???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??(?3$??I??B??mQ@QG{?F>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?lY?.p@?lY?.p@!?lY?.p@      ??!       "	?=?U?\@?=?U?\@!?=?U?\@*      ??!       2	?)??Y????)??Y???!?)??Y???:	&??s|? @&??s|? @!&??s|? @B      ??!       J	̛õ?Þ?̛õ?Þ?!̛õ?Þ?R      ??!       Z	̛õ?Þ?̛õ?Þ?!̛õ?Þ?b      ??!       JGPUY??(?3$??b q??B??mQ@yG{?F>@