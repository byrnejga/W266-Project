?	?i??Lw@?i??Lw@!?i??Lw@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?i??Lw@
???k,o@1? ??%?\@AR?GT???I?)??%p@*	8?A`?<f@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceo?[t?Ԣ?!??ĸr?4@)o?[t?Ԣ?1??ĸr?4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateGtϺF˱?!`????C@)?B????1;@??e2@:Preprocessing2U
Iterator::Model::ParallelMapV2?K⬈???!???1}:2@)?K⬈???1???1}:2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?b?0???!??H'd?5@)??=??W??1LQ???1@:Preprocessing2F
Iterator::Model?ם?<???!e?"jM??@)ew?h???1??p?+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?O??n??!?Vw?lQ@)M0?k????1?G[*K[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1?|?!??K?Rz@)y?&1?|?1??K?Rz@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?_?????!????=?D@)???q??q?1???w?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 66.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?u%.AQ@Q?(?kG?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	
???k,o@
???k,o@!
???k,o@      ??!       "	? ??%?\@? ??%?\@!? ??%?\@*      ??!       2	R?GT???R?GT???!R?GT???:	?)??%p@?)??%p@!?)??%p@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?u%.AQ@y?(?kG?>@?"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin1/Tensordot/MatMul/MatMulMatMulV???B???!V???B???0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMul/MatMulMatMulx?e?!g??u??0"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin1/Tensordot/MatMul/MatMul_1MatMulg??|??!?"5???"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMul_1MatMul&?r?y???!þ??"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin1/Tensordot/MatMul/MatMul_1MatMulɔK?s=??!H??L"??"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMulMatMul?#??Ճ?!fFMf???0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMulMatMul~XpȲ҃?!vQ[?/???0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMul/MatMulMatMul?퍫?ƃ?!/?t???0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMul/MatMulMatMul?%쓡Ń?!??J??w??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMul/MatMulMatMul!DWŃ?!
vˏc???0Q      Y@Y??e?9??a?iƧ?X@q??PX@y?TOZ??c?"?

both?Your program is POTENTIALLY input-bound because 66.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 