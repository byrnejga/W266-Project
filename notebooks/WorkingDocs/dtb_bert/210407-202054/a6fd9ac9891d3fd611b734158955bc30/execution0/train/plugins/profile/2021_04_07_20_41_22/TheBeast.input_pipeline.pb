	i9?Cm?w@i9?Cm?w@!i9?Cm?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-i9?Cm?w@??F??to@1?????\@A??l??p??I??y?!@*	Y9??vN\@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate^??#n??!?Y?Q?@@)o??g??1T.?&';@:Preprocessing2F
Iterator::Model?N\?W ??!?z&??F@)???B??1ĕ Rv?7@:Preprocessing2U
Iterator::Model::ParallelMapV2???B????!?_??Ɏ5@)???B????1?_??Ɏ5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat!@??T??!??AA͝/@)?h8en??1?A8?"?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice6Y???}?!????@)6Y???}?1????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????jد?!S???_wK@)V?F?q?1`M?:U?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???W?h?!o%??z@)???W?h?1o%??z@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??R??q??!???5??A@)*??g\8`?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?a+??SQ@QNxRߝ?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??F??to@??F??to@!??F??to@      ??!       "	?????\@?????\@!?????\@*      ??!       2	??l??p????l??p??!??l??p??:	??y?!@??y?!@!??y?!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?a+??SQ@yNxRߝ?>@