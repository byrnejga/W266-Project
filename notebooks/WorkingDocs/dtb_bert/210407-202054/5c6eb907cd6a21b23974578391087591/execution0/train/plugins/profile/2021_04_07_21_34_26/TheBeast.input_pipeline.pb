	????xi@????xi@!????xi@	?X? >n??X? >n?!?X? >n?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6????xi@n????+c@1????#E@A???*?]??I?K?^ @Y??c?~?*	NbX9?]@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?B?????!)]?A@)?I??	١?1? ??77=@:Preprocessing2F
Iterator::Model3?`????!?!?&	E@)?????'??1x????96@:Preprocessing2U
Iterator::Model::ParallelMapV2~nh?N???!????z?3@)~nh?N???1????z?3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ڧ?1??! ?X?21@)~oӟ?H??1????kK,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??H?}}?!Z?X?8#@)??H?}}?1Z?X?8#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???????!r????L@)?_????s?1\?N??A@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor6Y???m?!@,,?g@)6Y???m?1@,,?g@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?$w?Df??!V?7?OUB@)?@??_?[?1*?M;????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 75.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?X? >n?I?ŭ???S@QW?G?K?4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	n????+c@n????+c@!n????+c@      ??!       "	????#E@????#E@!????#E@*      ??!       2	???*?]?????*?]??!???*?]??:	?K?^ @?K?^ @!?K?^ @B      ??!       J	??c?~???c?~?!??c?~?R      ??!       Z	??c?~???c?~?!??c?~?b      ??!       JGPUY?X? >n?b q?ŭ???S@yW?G?K?4@