	?>???x@?>???x@!?>???x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?>???x@?b?dq@1O?\?\@AB??K8???I?Z?kBr!@*	)\???,a@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateS?A?Ѫ??!?????E@)???????1cz^2?rB@:Preprocessing2U
Iterator::Model::ParallelMapV2C??A|`??!_>?H?0@)C??A|`??1_>?H?0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatw?Nyt#??!??盠?3@)????%??1X?k{/@:Preprocessing2F
Iterator::Modelc?????!f#h]??;@)9(a????1(T?)?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?J???ւ?!?
?{?@)?J???ւ?1?
?{?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?À%W??!'????R@)v??ݰm??1Ե???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorg??j+?w?!?*v??@)g??j+?w?1?*v??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2?F? ??!?낪??F@)a??+ei?1?2^?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 68.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??U??Q@QE?[?b(=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?b?dq@?b?dq@!?b?dq@      ??!       "	O?\?\@O?\?\@!O?\?\@*      ??!       2	B??K8???B??K8???!B??K8???:	?Z?kBr!@?Z?kBr!@!?Z?kBr!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??U??Q@yE?[?b(=@