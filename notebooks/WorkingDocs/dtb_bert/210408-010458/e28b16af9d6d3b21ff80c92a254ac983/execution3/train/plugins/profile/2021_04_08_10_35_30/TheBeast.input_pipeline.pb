	عi3γw@عi3γw@!عi3γw@	e?H*L??e?H*L??!e?H*L??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6عi3γw@m????o@1??fG?]@A?$xC??I?+f??@Y+?6+1??*	}?5^?1b@2F
Iterator::Model!??i??!k7??pF@)gs?6??1M?w??7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?΅?^Ԧ?!?zDM?>@)?7? ???1v?OҤ6@:Preprocessing2U
Iterator::Model::ParallelMapV2Y?_"?:??!?s???4@)Y?_"?:??1?s???4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatIg`?eM??!]??#?2@)?TQ??ږ?1????.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceoG8-xч?!?y????@)oG8-xч?1?y????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?1?#ٴ?!??9-??K@)?#K?x?1o????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor
K<?l?u?!??q=@)
K<?l?u?1??q=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????n??!ؼ?gd@@)N^??i?1G??[4@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9e?H*L??Ib?%=?SQ@Q^`?j?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	m????o@m????o@!m????o@      ??!       "	??fG?]@??fG?]@!??fG?]@*      ??!       2	?$xC???$xC??!?$xC??:	?+f??@?+f??@!?+f??@B      ??!       J	+?6+1??+?6+1??!+?6+1??R      ??!       Z	+?6+1??+?6+1??!+?6+1??b      ??!       JGPUYe?H*L??b qb?%=?SQ@y^`?j?>@