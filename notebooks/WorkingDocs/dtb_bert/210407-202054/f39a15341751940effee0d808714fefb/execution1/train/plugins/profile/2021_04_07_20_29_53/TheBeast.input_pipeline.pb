	?1<???x@?1<???x@!?1<???x@	A??AwU??A??AwU??!A??AwU??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?1<???x@.????p@1K ?)U?\@A#??]???I??ܚt"@Y?J?ó??*	???S[b@2U
Iterator::Model::ParallelMapV2Q.?_x%??!???i??6@)Q.?_x%??1???i??6@:Preprocessing2F
Iterator::Model???????!?Be??F@)?^(`;??1??>J?6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??'Hlw??!??????4@)?aod??1?
la#?0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateo???ģ?!????mJ:@)s?]?????1:???m+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicel???????!???"')@)l???????1???"')@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip#?ng_y??!H??;?:K@)?M+?@.??1??e??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?UIddy?!?\???@)?UIddy?1?\???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapc?~?x???!r?-???;@);?O??nb?1??-垃??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 68.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9A??AwU??I???Q@Q?1r.k=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	.????p@.????p@!.????p@      ??!       "	K ?)U?\@K ?)U?\@!K ?)U?\@*      ??!       2	#??]???#??]???!#??]???:	??ܚt"@??ܚt"@!??ܚt"@B      ??!       J	?J?ó???J?ó??!?J?ó??R      ??!       Z	?J?ó???J?ó??!?J?ó??b      ??!       JGPUYA??AwU??b q???Q@y?1r.k=@