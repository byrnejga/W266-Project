	???֛w@???֛w@!???֛w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???֛w@?k????o@1-x?W?]@A??1v?K??I}????v"@*	?(\??!b@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateR??8ӄ??!??[?z?<@)?K⬈???1?熘9[6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV?j-?B??!?L	V.?9@)(??G???1??u?u5@:Preprocessing2F
Iterator::Model?.PR`??!/?????B@)????H???1·VF3@:Preprocessing2U
Iterator::Model::ParallelMapV2?h?wa??!Mi??o2@)?h?wa??1Mi??o2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea2U0*???!?PT?y@)a2U0*???1?PT?y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????!??!??@UQ%O@)?8?ߡ(??1L???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?BB?z?!8 ????@)?BB?z?18 ????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap ??*Q???!?? O??>@)?j??g?1W<Nj+??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??Ak?QQ@QsY?R6?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?k????o@?k????o@!?k????o@      ??!       "	-x?W?]@-x?W?]@!-x?W?]@*      ??!       2	??1v?K????1v?K??!??1v?K??:	}????v"@}????v"@!}????v"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??Ak?QQ@ysY?R6?>@