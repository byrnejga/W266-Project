	k??q??w@k??q??w@!k??q??w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-k??q??w@2s?ˣp@1??????\@ANa??????I?}?? @*	E?l???^@2F
Iterator::Model????[??!1???+?H@)?׃I????1"-U{iI<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!"a.(>@)???P??1?=ŋ߭8@:Preprocessing2U
Iterator::Model::ParallelMapV2?i? ?Ӛ?!?u*??$5@)?i? ?Ӛ?1?u*??$5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?\7??V??!???4K?,@)?R?{/??1??t?e?'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice겘?|\{?!"??q??@)겘?|\{?1"??q??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipI0??Z
??!?.?S?HI@)N^??y?1V]?_x5@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????i?!qZ???w@)??????i?1qZ???w@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????Դ??!??D?$?@)׆?q?&T?1??͢???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?? ]-jQ@Q??}?JW>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	2s?ˣp@2s?ˣp@!2s?ˣp@      ??!       "	??????\@??????\@!??????\@*      ??!       2	Na??????Na??????!Na??????:	?}?? @?}?? @!?}?? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?? ]-jQ@y??}?JW>@