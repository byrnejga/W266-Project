?	?=δ?x@?=δ?x@!?=δ?x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?=δ?x@c??ޚ?o@1n4??@9`@AN?=??j??I??}?u?"@*	J+??d@2F
Iterator::Model??'*ְ?!8?3H??C@)W??U???1xkPUG6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateǄ?K????!.J?9?H@@)ZI+?????1?&???5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????????!>?b06@)V???5??1???K?V2@:Preprocessing2U
Iterator::Model::ParallelMapV2,?z??m??!??S@?J1@),?z??m??1??S@?J1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?v????!??Rw?&@)?v????1??Rw?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip(?N>=???!?K̷7N@)????%?}?1%C2>@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-C??6z?!d??M6?@)-C??6z?1d??M6?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapςP??Ѭ?!D},???@@)??+ٱa?1?b?t1???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 64.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI=??S??P@Q???X?p@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	c??ޚ?o@c??ޚ?o@!c??ޚ?o@      ??!       "	n4??@9`@n4??@9`@!n4??@9`@*      ??!       2	N?=??j??N?=??j??!N?=??j??:	??}?u?"@??}?u?"@!??}?u?"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q=??S??P@y???X?p@@?"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin1/Tensordot/MatMul/MatMulMatMulx?/???!x?/???0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMul/MatMulMatMulbr????!???E*]??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMulMatMull?-Wr???!?f??qҤ?0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMul/MatMulMatMul??'?O??!???bf??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin1/Tensordot/MatMul/MatMulMatMulغ?]??!:??D?ް?0"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMul_1MatMul?`???I??!J?ٶ??"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMul/MatMul_1MatMul1Ξ??>??!??????"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMul_1MatMul?J!7= ??!k??|???"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMulMatMul.?1tr??!,B7???0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMulMatMul???3ii??!???)d/??0Q      Y@Y??e?9??a?iƧ?X@q????X@yg??`(nh?"?

both?Your program is POTENTIALLY input-bound because 64.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?97.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 