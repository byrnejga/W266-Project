	7???y@7???y@!7???y@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-7???y@?#??rmp@1 s-Z?-`@A???'???I??S?!@*	??Q?"`@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??M~?N??!???.?@@)t]???ԡ?16?L?:@:Preprocessing2U
Iterator::Model::ParallelMapV2f?"??)??!k)??t8@)f?"??)??1k)??t8@:Preprocessing2F
Iterator::Model???????!; ??D@)???aNЖ?1?LW?XB1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???????!?)?(?K3@)????0???1G}?)l0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceV?F???!??:@)V?F???1??:@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??rf?B??!????u$M@)?g??s?u?1s"?\l@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?aodn?!nc]&?@)?aodn?1nc]&?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???????!??[??pA@)?c#??W?1~??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??"?M?P@Q?κ?d!@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?#??rmp@?#??rmp@!?#??rmp@      ??!       "	 s-Z?-`@ s-Z?-`@! s-Z?-`@*      ??!       2	???'??????'???!???'???:	??S?!@??S?!@!??S?!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??"?M?P@y?κ?d!@@