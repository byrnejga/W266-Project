	h?o}?>i@h?o}?>i@!h?o}?>i@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-h?o}?>i@??˶?c@1?O??0kD@A?e???~??Iő"?? @*	h??|?M]@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate5]Ot]???!6U???=@)?g????1?????Y7@:Preprocessing2U
Iterator::Model::ParallelMapV2?st???!???@6@)?st???1???@6@:Preprocessing2F
Iterator::Model??B۩?!??7u??E@)?(yu???1??i,??4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?E|'f???!?'?V??4@)?i??&k??1?v?<1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?J??q??!Ct>b?`@)?J??q??1Ct>b?`@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipq?a????!J;ȊuL@)?+e?Xw?1ɂ?G?s@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor~oӟ?Hq?!E?%???@)~oӟ?Hq?1E?%???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ص?ݢ?!.??:p?@)y?&1?\?1-???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 75.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIP?-???S@Q?RI\84@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??˶?c@??˶?c@!??˶?c@      ??!       "	?O??0kD@?O??0kD@!?O??0kD@*      ??!       2	?e???~???e???~??!?e???~??:	ő"?? @ő"?? @!ő"?? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qP?-???S@y?RI\84@