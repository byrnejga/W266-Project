	?}?ƃwh@?}?ƃwh@!?}?ƃwh@	`a|?8L??`a|?8L??!`a|?8L??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?}?ƃwh@Ѳ?Jb@1??\4d?D@Aj??{???IJ+??@Ya??w}???*	?????m]@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?*O ???!wb!???@)???M~???1??0?V9@:Preprocessing2F
Iterator::Model/?혺+??!??K?j?F@)؀q???1? ???7@:Preprocessing2U
Iterator::Model::ParallelMapV2]lZ)r??!{????5@)]lZ)r??1{????5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat? #????!?TE?>?0@)??8?#+??1{Ln?)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice~t??gy~?!?uC??G@)~t??gy~?1?uC??G@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}v?uŌ??!\?C?uK@)Ƣ??dpt?1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorƊL??q?!??=?@)ƊL??q?1??=?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapV(??????!?ڑ??A@)^?/??f?1Q?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 74.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9`a|?8L??I??J?ޔS@Q?o?Sk5@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ѳ?Jb@Ѳ?Jb@!Ѳ?Jb@      ??!       "	??\4d?D@??\4d?D@!??\4d?D@*      ??!       2	j??{???j??{???!j??{???:	J+??@J+??@!J+??@B      ??!       J	a??w}???a??w}???!a??w}???R      ??!       Z	a??w}???a??w}???!a??w}???b      ??!       JGPUY`a|?8L??b q??J?ޔS@y?o?Sk5@