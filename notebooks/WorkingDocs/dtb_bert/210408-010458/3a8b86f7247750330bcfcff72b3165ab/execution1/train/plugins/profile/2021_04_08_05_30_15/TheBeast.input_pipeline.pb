	??Zz?w@??Zz?w@!??Zz?w@	5_p?A>?5_p?A>?!5_p?A>?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??Zz?w@???V<p@1ٙB?5?\@A$???+??I?/??#"@Y?6?h????*	?/?$_@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatel?衶??!?ۀQA@)ǂ L???1?+????;@:Preprocessing2U
Iterator::Model::ParallelMapV2#??u???!???T??:@)#??u???1???T??:@:Preprocessing2F
Iterator::Model?f?|???!??r?n~F@)???{,??1?Q?-<2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat`u?Hg`??!6??? 	0@)ȴ6?????1????L*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??hUM??!?.??U?@)??hUM??1?.??U?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip]?z??!0?\??K@)???s??r?1??L??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?n??Sm?!???93@)?n??Sm?1???93@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap1??c?g??!??(N??A@)??_vOV?1?>??g??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no96_p?A>?Ia"?G?Q@Qx?????=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???V<p@???V<p@!???V<p@      ??!       "	ٙB?5?\@ٙB?5?\@!ٙB?5?\@*      ??!       2	$???+??$???+??!$???+??:	?/??#"@?/??#"@!?/??#"@B      ??!       J	?6?h?????6?h????!?6?h????R      ??!       Z	?6?h?????6?h????!?6?h????b      ??!       JGPUY6_p?A>?b qa"?G?Q@yx?????=@