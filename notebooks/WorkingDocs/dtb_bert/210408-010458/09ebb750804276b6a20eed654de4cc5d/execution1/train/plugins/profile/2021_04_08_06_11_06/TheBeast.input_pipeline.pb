	?C?.?&x@?C?.?&x@!?C?.?&x@		??di??	??di??!	??di??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?C?.?&x@??neI+p@1[{?]@A)[$?F??I???ǵ?"@Y??]?????*	????? `@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????Li??!?_?XU@@)??'?8??1?3???7@:Preprocessing2U
Iterator::Model::ParallelMapV2fj?!???!?????@4@)fj?!???1?????@4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?:???!:u?*??6@)???r۾??1U_???2@:Preprocessing2F
Iterator::Model?Ŧ?B ??!]???A@)>!;oc???1?'?.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceܝ??.4??!IP?q?!@)ܝ??.4??1IP?q?!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipӢ>?6??!Q?y
.P@)?=~o??1ũ?B?F@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?}:3Py?!?W(?O@)?}:3Py?1?W(?O@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?S??Yh??!??/A??A@)![????o?1?y?.]@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9	??di??I5????WQ@QYL?G?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??neI+p@??neI+p@!??neI+p@      ??!       "	[{?]@[{?]@![{?]@*      ??!       2	)[$?F??)[$?F??!)[$?F??:	???ǵ?"@???ǵ?"@!???ǵ?"@B      ??!       J	??]???????]?????!??]?????R      ??!       Z	??]???????]?????!??]?????b      ??!       JGPUY	??di??b q5????WQ@yYL?G?>@