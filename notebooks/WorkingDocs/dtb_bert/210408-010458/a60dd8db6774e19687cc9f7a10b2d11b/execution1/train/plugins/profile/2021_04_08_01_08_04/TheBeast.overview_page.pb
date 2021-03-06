?	?x}j@?x}j@!?x}j@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?x}j@?mU?c@1+?w?7?D@A???!o??I?????U@*	?~j?t?g@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ؗl<ض?!???=G@)???GS??1???3?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??l#???!??????4@)???8?j??1?Q?D?1@:Preprocessing2F
Iterator::ModelL???j???!?????-:@)???̚?16?P?C+@:Preprocessing2U
Iterator::Model::ParallelMapV2???lɪ??!?9?YM)@)???lɪ??1?9?YM)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?f????!?????u&@)?f????1?????u&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Fx$??!?????tR@)'?ei????1?U)??:@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????y?!?^??A
@)??????y?1?^??A
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???L0???!d? ?H@)???|~h?1tz8C;???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 76.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIO\?c?T@QĎ?q??3@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?mU?c@?mU?c@!?mU?c@      ??!       "	+?w?7?D@+?w?7?D@!+?w?7?D@*      ??!       2	???!o?????!o??!???!o??:	?????U@?????U@!?????U@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qO\?c?T@yĎ?q??3@?
"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMulMatMul+E?z	??!+E?z	??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMulMatMul׻p5????!? ?!??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMulMatMul[Ԡ????!? ??q@??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMulMatMul푳???!\A,????0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMulMatMul??_????!??x?J]??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMulMatMul?-?P????!DB?];??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._0/ffn/lin1/Tensordot/MatMulMatMul?sߊ՝?!???????0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._1/ffn/lin1/Tensordot/MatMulMatMulZ&?Ν?!Zn?۫???0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMulMatMul?C???ȝ?!i;N?_4??0"?
etf_distil_bert_for_sequence_classification/distilbert/transformer/layer_._3/ffn/lin1/Tensordot/MatMulMatMul?C???ȝ?!???????0Q      Y@Y???:
@aXG??).X@qPQ?ܷW@y?????>??"?
both?Your program is POTENTIALLY input-bound because 76.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?94.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 