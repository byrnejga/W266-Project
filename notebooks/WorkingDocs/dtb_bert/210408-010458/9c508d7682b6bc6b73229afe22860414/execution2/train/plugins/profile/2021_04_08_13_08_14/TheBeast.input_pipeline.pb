	?wDE?w@?wDE?w@!?wDE?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?wDE?w@u?B??o@1tA}˜?]@A?ʢ?????I??M~?n!@*	NbX9?_@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate1~??7??!?U?i??>@)?b?: ???1(y?6R?7@:Preprocessing2F
Iterator::Model&??????!??p٠lD@)#???R??1P????5@:Preprocessing2U
Iterator::Model::ParallelMapV2(~??k	??!?????*3@)(~??k	??1?????*3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatUL??pv??!5?k5@);:?Fv???1??/?0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice]3?f???!Qs3?dA@)]3?f???1Qs3?dA@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Ac&Q??!w	?&_?M@)???3.|?1?2ɹ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorkH?c?Cw?!ׯ[?u?@)kH?c?Cw?1ׯ[?u?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap&W??Ma??!kѹ^@@)P?mp?b?1? ???u??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??S.:Q@Q?P?F?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	u?B??o@u?B??o@!u?B??o@      ??!       "	tA}˜?]@tA}˜?]@!tA}˜?]@*      ??!       2	?ʢ??????ʢ?????!?ʢ?????:	??M~?n!@??M~?n!@!??M~?n!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??S.:Q@y?P?F?@