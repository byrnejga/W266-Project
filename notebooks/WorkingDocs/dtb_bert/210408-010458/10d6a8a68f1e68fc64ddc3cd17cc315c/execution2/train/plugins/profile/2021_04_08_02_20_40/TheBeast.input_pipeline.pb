	?S?ɢ?w@?S?ɢ?w@!?S?ɢ?w@	??G??/????G??/??!??G??/??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?S?ɢ?w@?wcA?p@1)[$?F]@AR??m???Ie??k]$@Y??iܛ???*	??ʡ?`@2F
Iterator::Model???)????!??8v?K@)?c??1??1EZ??3<@:Preprocessing2U
Iterator::Model::ParallelMapV2H?9????!?ue`;@)H?9????1?ue`;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???ܡ?!?g	??:@)#,*?t???1y?>Y¹5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatDQ?O?I??!????A?*@)V-???1??/??%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceO崧??x?!?I*?@)O崧??x?1?I*?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????;??!ǉ?5F@)j>"?Dr?1W?v??
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?y?Cn?k?!݇i?8@)?y?Cn?k?1݇i?8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???-΢?!s_??b?;@)?????%^?1,x??%??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??G??/??IpD2??eQ@Qu???1Y>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?wcA?p@?wcA?p@!?wcA?p@      ??!       "	)[$?F]@)[$?F]@!)[$?F]@*      ??!       2	R??m???R??m???!R??m???:	e??k]$@e??k]$@!e??k]$@B      ??!       J	??iܛ?????iܛ???!??iܛ???R      ??!       Z	??iܛ?????iܛ???!??iܛ???b      ??!       JGPUY??G??/??b qpD2??eQ@yu???1Y>@