?	{??{Ufx@{??{Ufx@!{??{Ufx@	?V???????V??????!?V??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6{??{Ufx@}?Ж?zo@1?,???>`@A?c\qqT??Is߉Y? @Y???H.??*	?Q???]@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?<e5]O??!`?q?#?@@)??=??W??1?4ZMx?:@:Preprocessing2U
Iterator::Model::ParallelMapV2^??jGq??!??<n??8@)^??jGq??1??<n??8@:Preprocessing2F
Iterator::Model/PR`L??!|3??ÞD@)??U?&??1O?-m0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?F???!??,bc2@)???s??1??J+.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?-:Yj??!?X$&>?@)?-:Yj??1?X$&>?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??????!??J?<aM@);?*??y?14?c?(?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??-?lp?!7<\??
@)??-?lp?17<\??
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?V%?}???!??%??A@)n??d?1??c?#\ @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 64.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?V??????Ib??ڪP@Qef??ޤ@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	}?Ж?zo@}?Ж?zo@!}?Ж?zo@      ??!       "	?,???>`@?,???>`@!?,???>`@*      ??!       2	?c\qqT???c\qqT??!?c\qqT??:	s߉Y? @s߉Y? @!s߉Y? @B      ??!       J	???H.?????H.??!???H.??R      ??!       Z	???H.?????H.??!???H.??b      ??!       JGPUY?V??????b qb??ڪP@yef??ޤ@@?"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin1/Tensordot/MatMul/MatMulMatMulq??????!q??????0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMul/MatMulMatMul??гt??!????????0"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMul/MatMul_1MatMul???O)???!n!_?* ??"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMul_1MatMul?F?????!&3e??j??"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin1/Tensordot/MatMul/MatMulMatMul3 NN????!?]|-?ɰ?0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMul/MatMulMatMul?KX????!gG?l???0"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMul_1MatMulr?q??a??!??u?????"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin1/Tensordot/MatMul/MatMul_1MatMul	?M?x{??!|N??w??"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMul/MatMul_1MatMulk?Ws??!i??̓E??"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMulMatMulxRH??j??!????Ҿ?0Q      Y@Y??e?9??a?iƧ?X@q6???V@y?f???^e?"?

both?Your program is POTENTIALLY input-bound because 64.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?91.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 