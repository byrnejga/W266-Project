	??????w@??????w@!??????w@	a??????a??????!a??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??????w@?%??%p@1???j?\@A$~?.r??I?~?x??@Yfj?!???*	-???d@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat ??^EF??!5<r?M<@)??8*7Q??1C?7?9}7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateo??g??!q@??C@)4???5??1??o??[7@:Preprocessing2U
Iterator::Model::ParallelMapV2?mP?????! ?????-@)?mP?????1 ?????-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicev?d??!?iB??-@)v?d??1?iB??-@:Preprocessing2F
Iterator::ModelX歺դ?!??f?T9@)?^zo??1???$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?P?f???!??I?ʪR@)=dʇ?j??12*;?i?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J??q??!?G??_?@)?J??q??1?G??_?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??T?????!????D@)??????i?1??
X????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9a??????I??˸?cQ@Q'y??m>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?%??%p@?%??%p@!?%??%p@      ??!       "	???j?\@???j?\@!???j?\@*      ??!       2	$~?.r??$~?.r??!$~?.r??:	?~?x??@?~?x??@!?~?x??@B      ??!       J	fj?!???fj?!???!fj?!???R      ??!       Z	fj?!???fj?!???!fj?!???b      ??!       JGPUYa??????b q??˸?cQ@y'y??m>@