	At??Sx@At??Sx@!At??Sx@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-At??Sx@?p!?`?o@1??}??_@A^??N??IH6W?sl @*	S????c@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateԙ{H?ޯ?!?u???C@)?^Cp\ƥ?1d?lπ;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Y?Nܣ?!b?]A9?8@)
?_??͠?1???z?4@:Preprocessing2U
Iterator::Model::ParallelMapV2j/?혺??!	???;:1@)j/?혺??1	???;:1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?up?71??!C
d)@)?up?71??1C
d)@:Preprocessing2F
Iterator::Modell@??r???!??(?Y?<@)nQf?L2??1wV?<?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipA?)V¼?!??u???Q@):̗`}?1Y?sq?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor~?<?rx?!Þ83&a@)~?<?rx?1Þ83&a@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?.6???!?7?9"E@)a???)q?1c!???S@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noId'2m??P@Q7??%]f@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?p!?`?o@?p!?`?o@!?p!?`?o@      ??!       "	??}??_@??}??_@!??}??_@*      ??!       2	^??N??^??N??!^??N??:	H6W?sl @H6W?sl @!H6W?sl @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qd'2m??P@y7??%]f@@