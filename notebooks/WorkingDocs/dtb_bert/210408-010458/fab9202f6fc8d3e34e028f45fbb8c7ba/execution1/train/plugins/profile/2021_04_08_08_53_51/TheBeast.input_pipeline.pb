	??Z???w@??Z???w@!??Z???w@	?_L%<֐??_L%<֐?!?_L%<֐?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??Z???w@O?}?;p@1??W?2?\@AZc?	????Iol?`?!@Y??*?)??*	P??n?b@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?y?Տ??!?yhe(C@)??&?E'??1???Y??8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?fh??!L˒?V:@)j?~?^???1?n?W??5@:Preprocessing2U
Iterator::Model::ParallelMapV2.9????!?t??v,@).9????1?t??v,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?;? є?!?	??@?*@)?;? є?1?	??@?*@:Preprocessing2F
Iterator::Model??5&Ĥ?!??U???:@)ޯ|?y??1?V?_")@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziprk?m?\??!?????LR@)]?`7l[??1??[E@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorG=D?;?}?!?q??@@)G=D?;?}?1?q??@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????9??!??P&D@)9??v??j?1????-@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?_L%<֐?I0???3|Q@Q(l+??
>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	O?}?;p@O?}?;p@!O?}?;p@      ??!       "	??W?2?\@??W?2?\@!??W?2?\@*      ??!       2	Zc?	????Zc?	????!Zc?	????:	ol?`?!@ol?`?!@!ol?`?!@B      ??!       J	??*?)????*?)??!??*?)??R      ??!       Z	??*?)????*?)??!??*?)??b      ??!       JGPUY?_L%<֐?b q0???3|Q@y(l+??
>@