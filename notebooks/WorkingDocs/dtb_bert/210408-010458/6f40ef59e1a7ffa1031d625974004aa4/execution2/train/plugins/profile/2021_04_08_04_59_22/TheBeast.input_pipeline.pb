	??Fu?5x@??Fu?5x@!??Fu?5x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??Fu?5x@B???jp@1???O]@A??w???I&??)L!@*	?E???b@2F
Iterator::Model??
????!L?6??J@)t???)??1??X?;@:Preprocessing2U
Iterator::Model::ParallelMapV2?LnY??!??P(+:@)?LnY??1??P(+:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZ???ZУ?!HL?z?:@)O;?5Y???1???(?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatG?&jin??!{??"??,@)]?@???1/'??n(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??!o????!M"b?JQ@)??!o????1M"b?JQ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipI?<?+J??!?y??bG@)R~R???x?1%??}?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?A?L??j?!7K??9@)?A?L??j?17K??9@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$??辤?!"EB <@)6Y???]?1?mcU*??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?*5??Q@QCU+???=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	B???jp@B???jp@!B???jp@      ??!       "	???O]@???O]@!???O]@*      ??!       2	??w?????w???!??w???:	&??)L!@&??)L!@!&??)L!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?*5??Q@yCU+???=@