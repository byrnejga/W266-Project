	?E??y@?E??y@!?E??y@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?E??y@?8?~dp@1=ڨN/`@A>???@??Ix)u?8>"@*	?p=
?a@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Ɵ?lX??!$$f??C@)?1??8??1?<???:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?c?3?%??!0?&Hf;@)a?4?͟?1՟??2?6@:Preprocessing2U
Iterator::Model::ParallelMapV2????&M??!?&?.\?+@)????&M??1?&?.\?+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????g???!??p`*@)????g???1??p`*@:Preprocessing2F
Iterator::Model?3?<F??!^=?ַ8@)X?L??~??1P??qQ?%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+?w?7N??!??K
?R@)??z?p̂?1r-??.?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????y?!?@??U?@)??????y?1?@??U?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapwf??\ì?!H?C?*?D@)??)??f?1H?ܝ?: @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI~8??P@Q?%??%@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?8?~dp@?8?~dp@!?8?~dp@      ??!       "	=ڨN/`@=ڨN/`@!=ڨN/`@*      ??!       2	>???@??>???@??!>???@??:	x)u?8>"@x)u?8>"@!x)u?8>"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q~8??P@y?%??%@@