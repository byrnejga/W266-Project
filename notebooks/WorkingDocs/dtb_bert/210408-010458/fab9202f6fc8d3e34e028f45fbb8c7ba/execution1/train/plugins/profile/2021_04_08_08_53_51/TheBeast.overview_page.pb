?	??Z???w@??Z???w@!??Z???w@	?_L%<֐??_L%<֐?!?_L%<֐?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??Z???w@O?}?;p@1??W?2?\@AZc?	????Iol?`?!@Y??*?)??*	P??n?b@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?y?Տ??!?yhe(C@)??&?E'??1???Y??8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?fh??!L˒?V:@)j?~?^???1?n?W??5@:Preprocessing2U
Iterator::Model::ParallelMapV2.9????!?t??v,@).9????1?t??v,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?;? є?!?	??@?*@)?;? є?1?	??@?*@:Preprocessing2F
Iterator::Model??5&Ĥ?!??U???:@)ޯ|?y??1?V?_")@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziprk?m?\??!?????LR@)]?`7l[??1??[E@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorG=D?;?}?!?q??@@)G=D?;?}?1?q??@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????9??!??P&D@)9??v??j?1????-@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?_L%<֐?I0???3|Q@Q(l+??
>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	O?}?;p@O?}?;p@!O?}?;p@      ??!       "	??W?2?\@??W?2?\@!??W?2?\@*      ??!       2	Zc?	????Zc?	????!Zc?	????:	ol?`?!@ol?`?!@!ol?`?!@B      ??!       J	??*?)????*?)??!??*?)??R      ??!       Z	??*?)????*?)??!??*?)??b      ??!       JGPUY?_L%<֐?b q0???3|Q@y(l+??
>@?"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMul/MatMulMatMul;G?f<3??!;G?f<3??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMul/MatMulMatMul?h??????!X?v????0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin1/Tensordot/MatMul/MatMulMatMulE?/j???!?V????0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMulMatMul??q5с??!F}^?o[??0"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMul/MatMul_1MatMul??gN????!?7|?A??"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMul/MatMulMatMulױp@?ڃ?!3N??c???0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMul/MatMulMatMulO?S??Ѓ?!}?#w6??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMulMatMul??a?ẽ?!??????0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin1/Tensordot/MatMul/MatMulMatMul ??s???!?nRC?!??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMul/MatMulMatMul??>ҷ???!?I?=	???0Q      Y@Y??e?9??a?iƧ?X@q'?3?W@y	 ާ?Mc?"?

both?Your program is POTENTIALLY input-bound because 67.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?92.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 