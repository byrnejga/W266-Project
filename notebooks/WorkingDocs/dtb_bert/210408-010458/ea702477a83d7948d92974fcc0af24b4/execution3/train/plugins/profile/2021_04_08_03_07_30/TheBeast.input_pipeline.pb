	&4I,)yw@&4I,)yw@!&4I,)yw@	?,e{????,e{???!?,e{???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6&4I,)yw@?m?ulo@1?&?E'?\@A??v????I9??U;!@Y?N??:7??*	䥛? ?d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;??l?Ѭ?!TՅ@?@@)??Za???1?=??;@:Preprocessing2F
Iterator::Model?`?HZ??!/?(??E@)=?E~???1?>/Y?n6@:Preprocessing2U
Iterator::Model::ParallelMapV2[A?+???!?'"?U?4@)[A?+???1?'"?U?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat!?rh????!/????1@)4?i?????1ZB?\??*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?d?F ^??!?b??I~@)?d?F ^??1?b??I~@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??t?(%??!?L??uhL@)r??Q??z?1ѥ?-9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?c#??w?!0٣?$@)?c#??w?10٣?$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$?w~Q???!?./k??A@)??#?k?10QC?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?,e{???II?O??OQ@Q???3??>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?m?ulo@?m?ulo@!?m?ulo@      ??!       "	?&?E'?\@?&?E'?\@!?&?E'?\@*      ??!       2	??v??????v????!??v????:	9??U;!@9??U;!@!9??U;!@B      ??!       J	?N??:7???N??:7??!?N??:7??R      ??!       Z	?N??:7???N??:7??!?N??:7??b      ??!       JGPUY?,e{???b qI?O??OQ@y???3??>@