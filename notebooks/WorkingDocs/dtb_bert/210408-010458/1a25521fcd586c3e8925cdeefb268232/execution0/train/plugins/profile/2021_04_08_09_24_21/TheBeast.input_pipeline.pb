	?kBZ??w@?kBZ??w@!?kBZ??w@	???֓?????֓??!???֓??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?kBZ??w@?1ZG??o@1UL??p]@A,?IEc???It?//??@Y??qQ-"??*t?V?f@)       =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate0??mP???!*??r+@@)?C?3???1:????6@:Preprocessing2F
Iterator::Model???^a???!?֫??&C@)-?????1??3Ҹ?3@:Preprocessing2U
Iterator::Model::ParallelMapV2?f??e??!2?#???2@)?f??e??12?#???2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??v1ͤ?!kV??o6@)?ٮ?ˠ?1-$?2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice}?b?: ??!jܱT?j#@)}?b?: ??1jܱT?j#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??pvk???!j)TG,?N@)'??@j??1????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?ݮ????!???KpK@)?ݮ????1???KpK@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap1Xr???!J?hA@)?HP?h?1??Q???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???֓??IpSֵ-TQ@Q??2d?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?1ZG??o@?1ZG??o@!?1ZG??o@      ??!       "	UL??p]@UL??p]@!UL??p]@*      ??!       2	,?IEc???,?IEc???!,?IEc???:	t?//??@t?//??@!t?//??@B      ??!       J	??qQ-"????qQ-"??!??qQ-"??R      ??!       Z	??qQ-"????qQ-"??!??qQ-"??b      ??!       JGPUY???֓??b qpSֵ-TQ@y??2d?>@