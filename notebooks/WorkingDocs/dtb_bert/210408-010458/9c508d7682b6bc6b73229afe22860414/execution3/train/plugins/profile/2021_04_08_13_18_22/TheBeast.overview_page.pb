?	3???~?w@3???~?w@!3???~?w@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-3???~?w@???~??o@1tE)!X?\@AU?g$B#??Iݱ?&!@*	j?t??g@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateڒUn2??!??????F@)?Y???а?1*?3Ŝ=A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? n/??!??E?v?6@)??rJ@L??1K????2@:Preprocessing2F
Iterator::Model??E????!V3?d?R8@),?z??m??1????+.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?`6???!??ñ&@)?`6???1??ñ&@:Preprocessing2U
Iterator::Model::ParallelMapV2W???x??!???	?y"@)W???x??1???	?y"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?)??s??!*s?&C?R@)
??ϛ???1 l??^@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorm??!?2????@)m??1?2????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?t?? ???!~`E#_?G@)	?^)?p?1n?7@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 67.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?l?McQ@Q??L??r>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???~??o@???~??o@!???~??o@      ??!       "	tE)!X?\@tE)!X?\@!tE)!X?\@*      ??!       2	U?g$B#??U?g$B#??!U?g$B#??:	ݱ?&!@ݱ?&!@!ݱ?&!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?l?McQ@y??L??r>@?"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin1/Tensordot/MatMul/MatMulMatMul?r[XV??!?r[XV??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMul/MatMulMatMulh?ּ|???!???????0"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin1/Tensordot/MatMul/MatMul_1MatMul?dL?~??!??_?w???"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMul/MatMulMatMul?????!?*N?;u??0"?
|gradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin1/Tensordot/MatMul/MatMul_1MatMul?N??E??!???M?F??"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMulMatMulۗ??ſ??!??O?J???0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMul/MatMulMatMul?Ȃ?ރ?!_??%7??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMul/MatMulMatMul???`ރ?!Rf,????0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMulMatMul-qR?׃?!x????-??0"?
zgradient_tape/tf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMul/MatMulMatMulޔ7<2???!?F?ҥ??0Q      Y@Y??e?9??a?iƧ?X@q??\	!X@y??x?g?"?

both?Your program is POTENTIALLY input-bound because 67.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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