?	??ډ??h@??ډ??h@!??ډ??h@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??ډ??h@?7????b@1?HV?D@A?f׽???IC?i?q
 @*	???S??f@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateiq?0'h??!???A??D@)???а??1S<9v?&?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???Or???!??䟃7@)˞6????1?['??#3@:Preprocessing2F
Iterator::Model#?tu?b??!??9J?Q=@)??|?r٠?1B?8;?	2@:Preprocessing2U
Iterator::Model::ParallelMapV2\?O???!A?|?&@)\?O???1A?|?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice)>>!;o??!?E@r?$@))>>!;o??1?E@r?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip ?yrM???!??q퓫Q@)??։ˁ?1?u?? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?? ?X4}?!e??=D@)?? ?X4}?1e??=D@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?I?p??!qB?BoE@)a2U0*?c?1?\!x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 75.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?l%$?S@Q??Mjo?4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?7????b@?7????b@!?7????b@      ??!       "	?HV?D@?HV?D@!?HV?D@*      ??!       2	?f׽????f׽???!?f׽???:	C?i?q
 @C?i?q
 @!C?i?q
 @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?l%$?S@y??Mjo?4@?
"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMulMatMul?2??#??!?2??#??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMulMatMul??c^??!b???0!??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMulMatMul?i?H"??!???W??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMulMatMul.?????!`>;??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMulMatMul?#fi??!???js??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMulMatMul?2R???!o?V??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin1/Tensordot/MatMulMatMulSboz???!M[}?U??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin1/Tensordot/MatMulMatMul?פ?p???!H??c???0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin1/Tensordot/MatMulMatMulU"?????!IM#?2H??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin1/Tensordot/MatMulMatMul??M)???!I???&??0Q      Y@Y???:
@aXG??).X@q??
?<?X@y?e??2f??"?
both?Your program is POTENTIALLY input-bound because 75.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?98.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 