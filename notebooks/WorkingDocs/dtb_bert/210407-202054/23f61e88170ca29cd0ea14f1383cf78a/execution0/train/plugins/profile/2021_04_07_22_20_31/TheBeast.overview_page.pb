?	!t?%aw@!t?%aw@!!t?%aw@	??8????8??!??8??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6!t?%aw@?)x?o@1?^Cp\?\@A4?"1???IϽ??#@Y.v??2S??*	?Vi@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??A?V???!.?Y>A@)>v()???1wE???8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??B?=??!-??8?6@)?!S>U??1q?'?-?2@:Preprocessing2F
Iterator::ModelE,b?aL??!?n??n?A@)֪]Ң?1?1?f?T2@:Preprocessing2U
Iterator::Model::ParallelMapV2??S?ơ?!???Z8P1@)??S?ơ?1???Z8P1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?؀q??!?}??"?$@)?؀q??1?}??"?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?P1?߄??!?ȩ??P@)9	?/????1M`?(?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??~??@?!?-?vZp@)??~??@?1?-?vZp@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM??΢w??! ?????A@)(?XQ?ih?1*???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??8??I????*CQ@Q	?D??>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)x?o@?)x?o@!?)x?o@      ??!       "	?^Cp\?\@?^Cp\?\@!?^Cp\?\@*      ??!       2	4?"1???4?"1???!4?"1???:	Ͻ??#@Ͻ??#@!Ͻ??#@B      ??!       J	.v??2S??.v??2S??!.v??2S??R      ??!       Z	.v??2S??.v??2S??!.v??2S??b      ??!       JGPUY??8??b q????*CQ@y	?D??>@?"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin1/Tensordot/MatMul/MatMulMatMul?+?O??!?+?O??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMul/MatMulMatMul?U??????!C?l?????0"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMul/MatMul_1MatMul??5???!????UҢ?"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin1/Tensordot/MatMul/MatMul_1MatMul?@?fY4??!?ol_??"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMul_1MatMul?1U??-??!ffļ骭?"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMul/MatMulMatMulֻ[????!????LR??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMulMatMul	UQ?????!??C?jγ?0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMul/MatMulMatMulB??F????!w?~J??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMul/MatMulMatMul?-xE?׃?!?|?sŸ?0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMul/MatMulMatMul?
??Ń?!([?>??0Q      Y@Y??e?9??a?iƧ?X@qi?2??T@y?9???qi?"?

both?Your program is POTENTIALLY input-bound because 66.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?83.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 