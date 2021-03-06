?	?}??Di@?}??Di@!?}??Di@	?Z??@????Z??@???!?Z??@???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?}??Di@?5=((?b@1?CQ?O?D@An?+????I??֥F? @Y7U?q7??*	u?V*d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??>????!?(?? vD@)7p??G??1ݘ?8/<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?8?Z???!??t?"%:@)???Oա?1??Nj??5@:Preprocessing2U
Iterator::Model::ParallelMapV2?_????!㷋6?R.@)?_????1㷋6?R.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?/?'??!?q?z)@)?/?'??1?q?z)@:Preprocessing2F
Iterator::ModelyW=`2??!??@\?9@)5?uX??1O??? %@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipl	??g???!??/???R@)˂???:??1b??N?G@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?i?*?~?!????8@)?i?*?~?1????8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?k????!Y3???/E@)???U+c?1?P??E5??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 75.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?Z??@???I????~?S@Q??M?4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?5=((?b@?5=((?b@!?5=((?b@      ??!       "	?CQ?O?D@?CQ?O?D@!?CQ?O?D@*      ??!       2	n?+????n?+????!n?+????:	??֥F? @??֥F? @!??֥F? @B      ??!       J	7U?q7??7U?q7??!7U?q7??R      ??!       Z	7U?q7??7U?q7??!7U?q7??b      ??!       JGPUY?Z??@???b q????~?S@y??M?4@?
"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMulMatMul??=_@ҟ?!??=_@ҟ?0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMulMatMulm??'????!|xrC????0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMulMatMul??D?5???!?s??????0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMulMatMul?}?:/ڞ?!m?6eO??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMulMatMul? Tp,֞?!?ͥ?P???0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMulMatMul?\U?s???!dyP@?Y??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin1/Tensordot/MatMulMatMulȋ:|w???!???/???0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin1/Tensordot/MatMulMatMulݐ?+:???!??Suu???0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMulMatMulZ??}l???!"???;??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin1/Tensordot/MatMulMatMul?L??˖??!??<~??0Q      Y@Y???:
@aXG??).X@q޺?
wX@y??B????"?
both?Your program is POTENTIALLY input-bound because 75.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 