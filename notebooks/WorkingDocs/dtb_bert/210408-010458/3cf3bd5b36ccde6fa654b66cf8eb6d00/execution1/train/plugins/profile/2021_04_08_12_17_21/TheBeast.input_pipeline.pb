	??IӠ?w@??IӠ?w@!??IӠ?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??IӠ?w@???wp@1??????\@A??5?ڋ??I\Va3?? @*+???Y@)      =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatE?ӻx???!??d??<@)??KqUٗ?1r9I?z6@:Preprocessing2U
Iterator::Model::ParallelMapV2?<HO?C??!?eĕ?3@)?<HO?C??1?eĕ?3@:Preprocessing2F
Iterator::ModelZ_&??!?fGـA@)~?k?,	??1?Δ9?:.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?k?˸??!f????>8@)?X?? ??1(?^O?G,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~?d?p??!?????5$@)?~?d?p??1?????5$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???|?r??!?L\??rP@)Q?|a2??1)?[??#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?*??y?!?2?P3 @);?*??y?1?2?P3 @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap#f?y????!q???H;@) ?o_?i?1^H(7?R@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIl%^/amQ@QNj?B{J>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???wp@???wp@!???wp@      ??!       "	??????\@??????\@!??????\@*      ??!       2	??5?ڋ????5?ڋ??!??5?ڋ??:	\Va3?? @\Va3?? @!\Va3?? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb ql%^/amQ@yNj?B{J>@