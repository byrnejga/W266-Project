?	I?H?i@I?H?i@!I?H?i@	?'??ؔ??'??ؔ?!?'??ؔ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6I?H?i@?mP?-?b@1%??E@A?K?b??I?,?cy@Y?8?Վ???*	?????`@2F
Iterator::Modelm?kA???!????o+K@)?O ?Ȣ?16t?n?t<@:Preprocessing2U
Iterator::Model::ParallelMapV2Ãf׽??!??R???9@)Ãf׽??1??R???9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateR}?%???!?{??+8@)7߈?Yט?1??w^??2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?B??˔?!??.?P?/@)	?/?????1????6(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicelxz?,C|?!?c?Deh@)lxz?,C|?1?c?Deh@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ՏM?#??!iW[??F@)??@??s?1L??ƍ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU???N@s?!????*@)U???N@s?1????*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapo??o?D??!?7ƛf):@)YLl>?e?1??K?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 75.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?'??ؔ?Im?g{?S@QE?YZ?5@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?mP?-?b@?mP?-?b@!?mP?-?b@      ??!       "	%??E@%??E@!%??E@*      ??!       2	?K?b???K?b??!?K?b??:	?,?cy@?,?cy@!?,?cy@B      ??!       J	?8?Վ????8?Վ???!?8?Վ???R      ??!       Z	?8?Վ????8?Վ???!?8?Վ???b      ??!       JGPUY?'??ؔ?b qm?g{?S@yE?YZ?5@?
"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMulMatMulǻ??y??!ǻ??y??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMulMatMul>?5?m??!'} ?s??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMulMatMul6???.k??!?<F?Զ?0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMulMatMul????md??!??v??m??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMulMatMul?6??c??!??m?P??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMulMatMul@W!`??!?ǘR???0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin1/Tensordot/MatMulMatMul*iGt<??!鴁?????0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin1/Tensordot/MatMulMatMul?dL?fS??!?A?rMA??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMulMatMul?Am?E??!?tl????0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin1/Tensordot/MatMulMatMul=D??C??!?? ?????0Q      Y@Y???:
@aXG??).X@q?$??CX@y( M?-??"?
both?Your program is POTENTIALLY input-bound because 75.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?97.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 