	4f?`i@4f?`i@!4f?`i@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-4f?`i@??S?% c@1??h:;?D@A??u?ݵ?I??+???@*	G?z??`@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate"??gx???!?.??m?@@)?????12???;;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatK??????!=?X?8@)Eh׿??1????[4@:Preprocessing2F
Iterator::Model/??[<???!?F?ܪiA@)?3?/.U??1??5???2@:Preprocessing2U
Iterator::Model::ParallelMapV2????J#??!~;??=0@)????J#??1~;??=0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??ek}???!r;&?OO@)??ek}???1r;&?OO@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipb????5??!?ܣ?*KP@)cz?({?1yTQ
1?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??@???x?!??].@)??@???x?1??].@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapeRC???!?*Q?H?A@)8?*5{?e?1???a???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 75.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?k?]??S@QQ???4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??S?% c@??S?% c@!??S?% c@      ??!       "	??h:;?D@??h:;?D@!??h:;?D@*      ??!       2	??u?ݵ???u?ݵ?!??u?ݵ?:	??+???@??+???@!??+???@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?k?]??S@yQ???4@