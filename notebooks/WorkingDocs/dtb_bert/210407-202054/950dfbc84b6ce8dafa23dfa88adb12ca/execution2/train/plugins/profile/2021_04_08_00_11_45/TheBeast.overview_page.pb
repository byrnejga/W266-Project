?	.s?Kx@.s?Kx@!.s?Kx@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-.s?Kx@?_??qp@1R???T2]@AB?%U?M??I'JB"m?!@*	???(\?_@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!g???oA@)?cZ?????1?zX??=@:Preprocessing2F
Iterator::Model?????ѩ?!????C@)?g?K6??1??(.?5@:Preprocessing2U
Iterator::Model::ParallelMapV2;?I/???!M(V?2@);?I/???1M(V?2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Y?e0??!<?*h1@)}?b?: ??1?4A??+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???:q9~?!?@S??,@)???:q9~?1?@S??,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?3ڪ$???!X??4N@) ????}?1???22?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorT????p?![o?P??	@)T????p?1[o?P??	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap5?磌??!?	D???B@)???Q??l?1PCָ(,@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIݟT?M}Q@Q????
>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?_??qp@?_??qp@!?_??qp@      ??!       "	R???T2]@R???T2]@!R???T2]@*      ??!       2	B?%U?M??B?%U?M??!B?%U?M??:	'JB"m?!@'JB"m?!@!'JB"m?!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qݟT?M}Q@y????
>@?"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin1/Tensordot/MatMul/MatMulMatMulGq???
??!Gq???
??0"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin1/Tensordot/MatMul/MatMul_1MatMul?A?>q??!?EH??"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMul/MatMul_1MatMul?/R?"??!???????"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMul_1MatMul??é???!?4?C???"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMul/MatMulMatMul?n??ۇ?!S?rF??0"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMul_1MatMul["z?O???!?Ll????"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMulMatMulE??ݧ??!?fM"?m??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMul/MatMulMatMul?7?b???!?H??8???0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMulMatMul?:K3????!<???T??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMul/MatMulMatMul5?tQ???!????Ƽ?0Q      Y@Y??e?9??a?iƧ?X@q?L??KX@y-8??t]k?"?

both?Your program is POTENTIALLY input-bound because 67.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?97.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 