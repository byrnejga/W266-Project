	j???x@j???x@!j???x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-j???x@?"??]3p@1-??2]@A??YKi??Il#??f~#@*	th??|Gf@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???[1??!TA?3?A@)??+?,??1???D7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??,z???!?4??8@)?e3????1?wZ?3@:Preprocessing2F
Iterator::Model?c???H??!泫T??@@)V-?(???1pR??1e3@:Preprocessing2U
Iterator::Model::ParallelMapV2?l??<+??!?*r???+@)?l??<+??1?*r???+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?,σ????!	l՝?(@)?,σ????1	l՝?(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip.Ui?k|??!&?U?P@)n?2d???1?!?
nA@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor :̗`?!4?R??0@) :̗`?14?R??0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\??J?H??!??????B@)??/?^|q?1H??L)@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??ewQ@Q?nQh">@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?"??]3p@?"??]3p@!?"??]3p@      ??!       "	-??2]@-??2]@!-??2]@*      ??!       2	??YKi????YKi??!??YKi??:	l#??f~#@l#??f~#@!l#??f~#@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??ewQ@y?nQh">@