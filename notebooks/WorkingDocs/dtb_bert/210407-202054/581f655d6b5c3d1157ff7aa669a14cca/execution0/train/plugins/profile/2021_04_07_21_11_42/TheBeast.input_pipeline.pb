	uV?1?w@uV?1?w@!uV?1?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-uV?1?w@?&1??n@1S?!?u^@A?k^?Y-??I????u?"@*	???(\a@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?f+/????!c??9mD@)????????1???%k?A@:Preprocessing2F
Iterator::Models???啫?!?4?0?C@)?G?)s???1??L_5@:Preprocessing2U
Iterator::Model::ParallelMapV2??!6X8??!&SRZ?1@)??!6X8??1&SRZ?1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?v稣??!???q8,@)???r-Z??1;"?LV'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice ????}?!??X?e@) ????}?1??X?e@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Q?=??!!??B?PN@)-C??6z?1d?ͷ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?&???Kj?!??bǮ?@)?&???Kj?1??bǮ?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapw,?IEc??!?f?,j?D@)(?XQ?iX?1?1/^~k??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI%??6Q@QkG(%???@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?&1??n@?&1??n@!?&1??n@      ??!       "	S?!?u^@S?!?u^@!S?!?u^@*      ??!       2	?k^?Y-???k^?Y-??!?k^?Y-??:	????u?"@????u?"@!????u?"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q%??6Q@ykG(%???@