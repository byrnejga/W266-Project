¡
í
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8©


embedding_60/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@2*(
shared_nameembedding_60/embeddings

+embedding_60/embeddings/Read/ReadVariableOpReadVariableOpembedding_60/embeddings*
_output_shapes
:	@2*
dtype0

conv1d_144/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2@*"
shared_nameconv1d_144/kernel
{
%conv1d_144/kernel/Read/ReadVariableOpReadVariableOpconv1d_144/kernel*"
_output_shapes
:2@*
dtype0
v
conv1d_144/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv1d_144/bias
o
#conv1d_144/bias/Read/ReadVariableOpReadVariableOpconv1d_144/bias*
_output_shapes
:@*
dtype0

conv1d_145/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2@*"
shared_nameconv1d_145/kernel
{
%conv1d_145/kernel/Read/ReadVariableOpReadVariableOpconv1d_145/kernel*"
_output_shapes
:2@*
dtype0
v
conv1d_145/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv1d_145/bias
o
#conv1d_145/bias/Read/ReadVariableOpReadVariableOpconv1d_145/bias*
_output_shapes
:@*
dtype0

conv1d_146/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2@*"
shared_nameconv1d_146/kernel
{
%conv1d_146/kernel/Read/ReadVariableOpReadVariableOpconv1d_146/kernel*"
_output_shapes
:2@*
dtype0
v
conv1d_146/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv1d_146/bias
o
#conv1d_146/bias/Read/ReadVariableOpReadVariableOpconv1d_146/bias*
_output_shapes
:@*
dtype0
}
dense_135/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À*!
shared_namedense_135/kernel
v
$dense_135/kernel/Read/ReadVariableOpReadVariableOpdense_135/kernel*
_output_shapes
:	À*
dtype0
t
dense_135/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_135/bias
m
"dense_135/bias/Read/ReadVariableOpReadVariableOpdense_135/bias*
_output_shapes
:*
dtype0
|
dense_136/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_136/kernel
u
$dense_136/kernel/Read/ReadVariableOpReadVariableOpdense_136/kernel*
_output_shapes

:*
dtype0
t
dense_136/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_136/bias
m
"dense_136/bias/Read/ReadVariableOpReadVariableOpdense_136/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/embedding_60/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@2*/
shared_name Adam/embedding_60/embeddings/m

2Adam/embedding_60/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_60/embeddings/m*
_output_shapes
:	@2*
dtype0

Adam/conv1d_144/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2@*)
shared_nameAdam/conv1d_144/kernel/m

,Adam/conv1d_144/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_144/kernel/m*"
_output_shapes
:2@*
dtype0

Adam/conv1d_144/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_144/bias/m
}
*Adam/conv1d_144/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_144/bias/m*
_output_shapes
:@*
dtype0

Adam/conv1d_145/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2@*)
shared_nameAdam/conv1d_145/kernel/m

,Adam/conv1d_145/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_145/kernel/m*"
_output_shapes
:2@*
dtype0

Adam/conv1d_145/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_145/bias/m
}
*Adam/conv1d_145/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_145/bias/m*
_output_shapes
:@*
dtype0

Adam/conv1d_146/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2@*)
shared_nameAdam/conv1d_146/kernel/m

,Adam/conv1d_146/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_146/kernel/m*"
_output_shapes
:2@*
dtype0

Adam/conv1d_146/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_146/bias/m
}
*Adam/conv1d_146/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_146/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_135/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À*(
shared_nameAdam/dense_135/kernel/m

+Adam/dense_135/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_135/kernel/m*
_output_shapes
:	À*
dtype0

Adam/dense_135/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_135/bias/m
{
)Adam/dense_135/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_135/bias/m*
_output_shapes
:*
dtype0

Adam/dense_136/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_136/kernel/m

+Adam/dense_136/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_136/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_136/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_136/bias/m
{
)Adam/dense_136/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_136/bias/m*
_output_shapes
:*
dtype0

Adam/embedding_60/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@2*/
shared_name Adam/embedding_60/embeddings/v

2Adam/embedding_60/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_60/embeddings/v*
_output_shapes
:	@2*
dtype0

Adam/conv1d_144/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2@*)
shared_nameAdam/conv1d_144/kernel/v

,Adam/conv1d_144/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_144/kernel/v*"
_output_shapes
:2@*
dtype0

Adam/conv1d_144/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_144/bias/v
}
*Adam/conv1d_144/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_144/bias/v*
_output_shapes
:@*
dtype0

Adam/conv1d_145/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2@*)
shared_nameAdam/conv1d_145/kernel/v

,Adam/conv1d_145/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_145/kernel/v*"
_output_shapes
:2@*
dtype0

Adam/conv1d_145/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_145/bias/v
}
*Adam/conv1d_145/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_145/bias/v*
_output_shapes
:@*
dtype0

Adam/conv1d_146/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2@*)
shared_nameAdam/conv1d_146/kernel/v

,Adam/conv1d_146/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_146/kernel/v*"
_output_shapes
:2@*
dtype0

Adam/conv1d_146/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_146/bias/v
}
*Adam/conv1d_146/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_146/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_135/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À*(
shared_nameAdam/dense_135/kernel/v

+Adam/dense_135/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_135/kernel/v*
_output_shapes
:	À*
dtype0

Adam/dense_135/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_135/bias/v
{
)Adam/dense_135/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_135/bias/v*
_output_shapes
:*
dtype0

Adam/dense_136/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_136/kernel/v

+Adam/dense_136/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_136/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_136/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_136/bias/v
{
)Adam/dense_136/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_136/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
±I
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ìH
valueâHBßH BØH
«
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
R
*	variables
+trainable_variables
,regularization_losses
-	keras_api
R
.	variables
/trainable_variables
0regularization_losses
1	keras_api
R
2	variables
3trainable_variables
4regularization_losses
5	keras_api
R
6	variables
7trainable_variables
8regularization_losses
9	keras_api
R
:	variables
;trainable_variables
<regularization_losses
=	keras_api
h

>kernel
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
h

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api

Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratemmmmm$m%m>m?mDmEm v¡v¢v£v¤v¥$v¦%v§>v¨?v©DvªEv«
N
0
1
2
3
4
$5
%6
>7
?8
D9
E10
N
0
1
2
3
4
$5
%6
>7
?8
D9
E10
 
­
	variables

Olayers
trainable_variables
Player_regularization_losses
Qlayer_metrics
Rnon_trainable_variables
Smetrics
regularization_losses
 
ge
VARIABLE_VALUEembedding_60/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
	variables

Tlayers
trainable_variables
Ulayer_metrics
Vlayer_regularization_losses
Wnon_trainable_variables
Xmetrics
regularization_losses
][
VARIABLE_VALUEconv1d_144/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv1d_144/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
	variables

Ylayers
trainable_variables
Zlayer_metrics
[layer_regularization_losses
\non_trainable_variables
]metrics
regularization_losses
][
VARIABLE_VALUEconv1d_145/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv1d_145/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
 	variables

^layers
!trainable_variables
_layer_metrics
`layer_regularization_losses
anon_trainable_variables
bmetrics
"regularization_losses
][
VARIABLE_VALUEconv1d_146/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv1d_146/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
­
&	variables

clayers
'trainable_variables
dlayer_metrics
elayer_regularization_losses
fnon_trainable_variables
gmetrics
(regularization_losses
 
 
 
­
*	variables

hlayers
+trainable_variables
ilayer_metrics
jlayer_regularization_losses
knon_trainable_variables
lmetrics
,regularization_losses
 
 
 
­
.	variables

mlayers
/trainable_variables
nlayer_metrics
olayer_regularization_losses
pnon_trainable_variables
qmetrics
0regularization_losses
 
 
 
­
2	variables

rlayers
3trainable_variables
slayer_metrics
tlayer_regularization_losses
unon_trainable_variables
vmetrics
4regularization_losses
 
 
 
­
6	variables

wlayers
7trainable_variables
xlayer_metrics
ylayer_regularization_losses
znon_trainable_variables
{metrics
8regularization_losses
 
 
 
®
:	variables

|layers
;trainable_variables
}layer_metrics
~layer_regularization_losses
non_trainable_variables
metrics
<regularization_losses
\Z
VARIABLE_VALUEdense_135/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_135/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1

>0
?1
 
²
@	variables
layers
Atrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
metrics
Bregularization_losses
\Z
VARIABLE_VALUEdense_136/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_136/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1

D0
E1
 
²
F	variables
layers
Gtrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
metrics
Hregularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
V
0
1
2
3
4
5
6
7
	8

9
10
11
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables

VARIABLE_VALUEAdam/embedding_60/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv1d_144/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d_144/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv1d_145/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d_145/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv1d_146/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d_146/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_135/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_135/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_136/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_136/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/embedding_60/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv1d_144/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d_144/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv1d_145/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d_145/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv1d_146/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d_146/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_135/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_135/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_136/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_136/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_input_61Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿd

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_61embedding_60/embeddingsconv1d_146/kernelconv1d_146/biasconv1d_145/kernelconv1d_145/biasconv1d_144/kernelconv1d_144/biasdense_135/kerneldense_135/biasdense_136/kerneldense_136/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1561263
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ë
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+embedding_60/embeddings/Read/ReadVariableOp%conv1d_144/kernel/Read/ReadVariableOp#conv1d_144/bias/Read/ReadVariableOp%conv1d_145/kernel/Read/ReadVariableOp#conv1d_145/bias/Read/ReadVariableOp%conv1d_146/kernel/Read/ReadVariableOp#conv1d_146/bias/Read/ReadVariableOp$dense_135/kernel/Read/ReadVariableOp"dense_135/bias/Read/ReadVariableOp$dense_136/kernel/Read/ReadVariableOp"dense_136/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp2Adam/embedding_60/embeddings/m/Read/ReadVariableOp,Adam/conv1d_144/kernel/m/Read/ReadVariableOp*Adam/conv1d_144/bias/m/Read/ReadVariableOp,Adam/conv1d_145/kernel/m/Read/ReadVariableOp*Adam/conv1d_145/bias/m/Read/ReadVariableOp,Adam/conv1d_146/kernel/m/Read/ReadVariableOp*Adam/conv1d_146/bias/m/Read/ReadVariableOp+Adam/dense_135/kernel/m/Read/ReadVariableOp)Adam/dense_135/bias/m/Read/ReadVariableOp+Adam/dense_136/kernel/m/Read/ReadVariableOp)Adam/dense_136/bias/m/Read/ReadVariableOp2Adam/embedding_60/embeddings/v/Read/ReadVariableOp,Adam/conv1d_144/kernel/v/Read/ReadVariableOp*Adam/conv1d_144/bias/v/Read/ReadVariableOp,Adam/conv1d_145/kernel/v/Read/ReadVariableOp*Adam/conv1d_145/bias/v/Read/ReadVariableOp,Adam/conv1d_146/kernel/v/Read/ReadVariableOp*Adam/conv1d_146/bias/v/Read/ReadVariableOp+Adam/dense_135/kernel/v/Read/ReadVariableOp)Adam/dense_135/bias/v/Read/ReadVariableOp+Adam/dense_136/kernel/v/Read/ReadVariableOp)Adam/dense_136/bias/v/Read/ReadVariableOpConst*7
Tin0
.2,	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1561785
	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_60/embeddingsconv1d_144/kernelconv1d_144/biasconv1d_145/kernelconv1d_145/biasconv1d_146/kernelconv1d_146/biasdense_135/kerneldense_135/biasdense_136/kerneldense_136/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/embedding_60/embeddings/mAdam/conv1d_144/kernel/mAdam/conv1d_144/bias/mAdam/conv1d_145/kernel/mAdam/conv1d_145/bias/mAdam/conv1d_146/kernel/mAdam/conv1d_146/bias/mAdam/dense_135/kernel/mAdam/dense_135/bias/mAdam/dense_136/kernel/mAdam/dense_136/bias/mAdam/embedding_60/embeddings/vAdam/conv1d_144/kernel/vAdam/conv1d_144/bias/vAdam/conv1d_145/kernel/vAdam/conv1d_145/bias/vAdam/conv1d_146/kernel/vAdam/conv1d_146/bias/vAdam/dense_135/kernel/vAdam/dense_135/bias/vAdam/dense_136/kernel/vAdam/dense_136/bias/v*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1561921î
å	

I__inference_embedding_60_layer_call_and_return_conditional_losses_1560846

inputs
embedding_lookup_1560840
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Castÿ
embedding_lookupResourceGatherembedding_lookup_1560840Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1560840*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
dtype02
embedding_lookupî
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1560840*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ö

,__inference_conv1d_144_layer_call_fn_1561504

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_144_layer_call_and_return_conditional_losses_15609382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd2::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
 
_user_specified_nameinputs
4
í
E__inference_model_60_layer_call_and_return_conditional_losses_1561137

inputs
embedding_60_1561103
conv1d_146_1561106
conv1d_146_1561108
conv1d_145_1561111
conv1d_145_1561113
conv1d_144_1561116
conv1d_144_1561118
dense_135_1561126
dense_135_1561128
dense_136_1561131
dense_136_1561133
identity¢"conv1d_144/StatefulPartitionedCall¢"conv1d_145/StatefulPartitionedCall¢"conv1d_146/StatefulPartitionedCall¢!dense_135/StatefulPartitionedCall¢!dense_136/StatefulPartitionedCall¢"dropout_60/StatefulPartitionedCall¢$embedding_60/StatefulPartitionedCall
$embedding_60/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_60_1561103*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_embedding_60_layer_call_and_return_conditional_losses_15608462&
$embedding_60/StatefulPartitionedCallÏ
"conv1d_146/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_146_1561106conv1d_146_1561108*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_146_layer_call_and_return_conditional_losses_15608742$
"conv1d_146/StatefulPartitionedCallÏ
"conv1d_145/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_145_1561111conv1d_145_1561113*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_145_layer_call_and_return_conditional_losses_15609062$
"conv1d_145/StatefulPartitionedCallÏ
"conv1d_144/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_144_1561116conv1d_144_1561118*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_144_layer_call_and_return_conditional_losses_15609382$
"conv1d_144/StatefulPartitionedCall­
(global_max_pooling1d_144/PartitionedCallPartitionedCall+conv1d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_144_layer_call_and_return_conditional_losses_15608002*
(global_max_pooling1d_144/PartitionedCall­
(global_max_pooling1d_145/PartitionedCallPartitionedCall+conv1d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_145_layer_call_and_return_conditional_losses_15608132*
(global_max_pooling1d_145/PartitionedCall­
(global_max_pooling1d_146/PartitionedCallPartitionedCall+conv1d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_146_layer_call_and_return_conditional_losses_15608262*
(global_max_pooling1d_146/PartitionedCallþ
concatenate_60/PartitionedCallPartitionedCall1global_max_pooling1d_144/PartitionedCall:output:01global_max_pooling1d_145/PartitionedCall:output:01global_max_pooling1d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_60_layer_call_and_return_conditional_losses_15609652 
concatenate_60/PartitionedCall
"dropout_60/StatefulPartitionedCallStatefulPartitionedCall'concatenate_60/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_60_layer_call_and_return_conditional_losses_15609872$
"dropout_60/StatefulPartitionedCallÄ
!dense_135/StatefulPartitionedCallStatefulPartitionedCall+dropout_60/StatefulPartitionedCall:output:0dense_135_1561126dense_135_1561128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_135_layer_call_and_return_conditional_losses_15610162#
!dense_135/StatefulPartitionedCallÃ
!dense_136/StatefulPartitionedCallStatefulPartitionedCall*dense_135/StatefulPartitionedCall:output:0dense_136_1561131dense_136_1561133*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_136_layer_call_and_return_conditional_losses_15610432#
!dense_136/StatefulPartitionedCall
IdentityIdentity*dense_136/StatefulPartitionedCall:output:0#^conv1d_144/StatefulPartitionedCall#^conv1d_145/StatefulPartitionedCall#^conv1d_146/StatefulPartitionedCall"^dense_135/StatefulPartitionedCall"^dense_136/StatefulPartitionedCall#^dropout_60/StatefulPartitionedCall%^embedding_60/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::2H
"conv1d_144/StatefulPartitionedCall"conv1d_144/StatefulPartitionedCall2H
"conv1d_145/StatefulPartitionedCall"conv1d_145/StatefulPartitionedCall2H
"conv1d_146/StatefulPartitionedCall"conv1d_146/StatefulPartitionedCall2F
!dense_135/StatefulPartitionedCall!dense_135/StatefulPartitionedCall2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2H
"dropout_60/StatefulPartitionedCall"dropout_60/StatefulPartitionedCall2L
$embedding_60/StatefulPartitionedCall$embedding_60/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
à

%__inference_signature_wrapper_1561263
input_61
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinput_61unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_15607932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
input_61
ö

,__inference_conv1d_146_layer_call_fn_1561554

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_146_layer_call_and_return_conditional_losses_15608742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd2::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
 
_user_specified_nameinputs
Ö2
È
E__inference_model_60_layer_call_and_return_conditional_losses_1561201

inputs
embedding_60_1561167
conv1d_146_1561170
conv1d_146_1561172
conv1d_145_1561175
conv1d_145_1561177
conv1d_144_1561180
conv1d_144_1561182
dense_135_1561190
dense_135_1561192
dense_136_1561195
dense_136_1561197
identity¢"conv1d_144/StatefulPartitionedCall¢"conv1d_145/StatefulPartitionedCall¢"conv1d_146/StatefulPartitionedCall¢!dense_135/StatefulPartitionedCall¢!dense_136/StatefulPartitionedCall¢$embedding_60/StatefulPartitionedCall
$embedding_60/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_60_1561167*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_embedding_60_layer_call_and_return_conditional_losses_15608462&
$embedding_60/StatefulPartitionedCallÏ
"conv1d_146/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_146_1561170conv1d_146_1561172*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_146_layer_call_and_return_conditional_losses_15608742$
"conv1d_146/StatefulPartitionedCallÏ
"conv1d_145/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_145_1561175conv1d_145_1561177*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_145_layer_call_and_return_conditional_losses_15609062$
"conv1d_145/StatefulPartitionedCallÏ
"conv1d_144/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_144_1561180conv1d_144_1561182*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_144_layer_call_and_return_conditional_losses_15609382$
"conv1d_144/StatefulPartitionedCall­
(global_max_pooling1d_144/PartitionedCallPartitionedCall+conv1d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_144_layer_call_and_return_conditional_losses_15608002*
(global_max_pooling1d_144/PartitionedCall­
(global_max_pooling1d_145/PartitionedCallPartitionedCall+conv1d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_145_layer_call_and_return_conditional_losses_15608132*
(global_max_pooling1d_145/PartitionedCall­
(global_max_pooling1d_146/PartitionedCallPartitionedCall+conv1d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_146_layer_call_and_return_conditional_losses_15608262*
(global_max_pooling1d_146/PartitionedCallþ
concatenate_60/PartitionedCallPartitionedCall1global_max_pooling1d_144/PartitionedCall:output:01global_max_pooling1d_145/PartitionedCall:output:01global_max_pooling1d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_60_layer_call_and_return_conditional_losses_15609652 
concatenate_60/PartitionedCall
dropout_60/PartitionedCallPartitionedCall'concatenate_60/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_60_layer_call_and_return_conditional_losses_15609922
dropout_60/PartitionedCall¼
!dense_135/StatefulPartitionedCallStatefulPartitionedCall#dropout_60/PartitionedCall:output:0dense_135_1561190dense_135_1561192*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_135_layer_call_and_return_conditional_losses_15610162#
!dense_135/StatefulPartitionedCallÃ
!dense_136/StatefulPartitionedCallStatefulPartitionedCall*dense_135/StatefulPartitionedCall:output:0dense_136_1561195dense_136_1561197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_136_layer_call_and_return_conditional_losses_15610432#
!dense_136/StatefulPartitionedCallÜ
IdentityIdentity*dense_136/StatefulPartitionedCall:output:0#^conv1d_144/StatefulPartitionedCall#^conv1d_145/StatefulPartitionedCall#^conv1d_146/StatefulPartitionedCall"^dense_135/StatefulPartitionedCall"^dense_136/StatefulPartitionedCall%^embedding_60/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::2H
"conv1d_144/StatefulPartitionedCall"conv1d_144/StatefulPartitionedCall2H
"conv1d_145/StatefulPartitionedCall"conv1d_145/StatefulPartitionedCall2H
"conv1d_146/StatefulPartitionedCall"conv1d_146/StatefulPartitionedCall2F
!dense_135/StatefulPartitionedCall!dense_135/StatefulPartitionedCall2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2L
$embedding_60/StatefulPartitionedCall$embedding_60/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ò	
ß
F__inference_dense_136_layer_call_and_return_conditional_losses_1561043

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´q
Ø	
"__inference__wrapped_model_1560793
input_612
.model_60_embedding_60_embedding_lookup_1560728C
?model_60_conv1d_146_conv1d_expanddims_1_readvariableop_resource7
3model_60_conv1d_146_biasadd_readvariableop_resourceC
?model_60_conv1d_145_conv1d_expanddims_1_readvariableop_resource7
3model_60_conv1d_145_biasadd_readvariableop_resourceC
?model_60_conv1d_144_conv1d_expanddims_1_readvariableop_resource7
3model_60_conv1d_144_biasadd_readvariableop_resource5
1model_60_dense_135_matmul_readvariableop_resource6
2model_60_dense_135_biasadd_readvariableop_resource5
1model_60_dense_136_matmul_readvariableop_resource6
2model_60_dense_136_biasadd_readvariableop_resource
identity¢*model_60/conv1d_144/BiasAdd/ReadVariableOp¢6model_60/conv1d_144/conv1d/ExpandDims_1/ReadVariableOp¢*model_60/conv1d_145/BiasAdd/ReadVariableOp¢6model_60/conv1d_145/conv1d/ExpandDims_1/ReadVariableOp¢*model_60/conv1d_146/BiasAdd/ReadVariableOp¢6model_60/conv1d_146/conv1d/ExpandDims_1/ReadVariableOp¢)model_60/dense_135/BiasAdd/ReadVariableOp¢(model_60/dense_135/MatMul/ReadVariableOp¢)model_60/dense_136/BiasAdd/ReadVariableOp¢(model_60/dense_136/MatMul/ReadVariableOp¢&model_60/embedding_60/embedding_lookup
model_60/embedding_60/CastCastinput_61*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
model_60/embedding_60/Castí
&model_60/embedding_60/embedding_lookupResourceGather.model_60_embedding_60_embedding_lookup_1560728model_60/embedding_60/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_60/embedding_60/embedding_lookup/1560728*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
dtype02(
&model_60/embedding_60/embedding_lookupÆ
/model_60/embedding_60/embedding_lookup/IdentityIdentity/model_60/embedding_60/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_60/embedding_60/embedding_lookup/1560728*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd221
/model_60/embedding_60/embedding_lookup/Identityâ
1model_60/embedding_60/embedding_lookup/Identity_1Identity8model_60/embedding_60/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd223
1model_60/embedding_60/embedding_lookup/Identity_1¡
)model_60/conv1d_146/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2+
)model_60/conv1d_146/conv1d/ExpandDims/dim
%model_60/conv1d_146/conv1d/ExpandDims
ExpandDims:model_60/embedding_60/embedding_lookup/Identity_1:output:02model_60/conv1d_146/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22'
%model_60/conv1d_146/conv1d/ExpandDimsô
6model_60/conv1d_146/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?model_60_conv1d_146_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype028
6model_60/conv1d_146/conv1d/ExpandDims_1/ReadVariableOp
+model_60/conv1d_146/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_60/conv1d_146/conv1d/ExpandDims_1/dim
'model_60/conv1d_146/conv1d/ExpandDims_1
ExpandDims>model_60/conv1d_146/conv1d/ExpandDims_1/ReadVariableOp:value:04model_60/conv1d_146/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2)
'model_60/conv1d_146/conv1d/ExpandDims_1
model_60/conv1d_146/conv1dConv2D.model_60/conv1d_146/conv1d/ExpandDims:output:00model_60/conv1d_146/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*
paddingVALID*
strides
2
model_60/conv1d_146/conv1dÎ
"model_60/conv1d_146/conv1d/SqueezeSqueeze#model_60/conv1d_146/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2$
"model_60/conv1d_146/conv1d/SqueezeÈ
*model_60/conv1d_146/BiasAdd/ReadVariableOpReadVariableOp3model_60_conv1d_146_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model_60/conv1d_146/BiasAdd/ReadVariableOpÜ
model_60/conv1d_146/BiasAddBiasAdd+model_60/conv1d_146/conv1d/Squeeze:output:02model_60/conv1d_146/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2
model_60/conv1d_146/BiasAdd
model_60/conv1d_146/ReluRelu$model_60/conv1d_146/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2
model_60/conv1d_146/Relu¡
)model_60/conv1d_145/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2+
)model_60/conv1d_145/conv1d/ExpandDims/dim
%model_60/conv1d_145/conv1d/ExpandDims
ExpandDims:model_60/embedding_60/embedding_lookup/Identity_1:output:02model_60/conv1d_145/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22'
%model_60/conv1d_145/conv1d/ExpandDimsô
6model_60/conv1d_145/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?model_60_conv1d_145_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype028
6model_60/conv1d_145/conv1d/ExpandDims_1/ReadVariableOp
+model_60/conv1d_145/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_60/conv1d_145/conv1d/ExpandDims_1/dim
'model_60/conv1d_145/conv1d/ExpandDims_1
ExpandDims>model_60/conv1d_145/conv1d/ExpandDims_1/ReadVariableOp:value:04model_60/conv1d_145/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2)
'model_60/conv1d_145/conv1d/ExpandDims_1
model_60/conv1d_145/conv1dConv2D.model_60/conv1d_145/conv1d/ExpandDims:output:00model_60/conv1d_145/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*
paddingVALID*
strides
2
model_60/conv1d_145/conv1dÎ
"model_60/conv1d_145/conv1d/SqueezeSqueeze#model_60/conv1d_145/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2$
"model_60/conv1d_145/conv1d/SqueezeÈ
*model_60/conv1d_145/BiasAdd/ReadVariableOpReadVariableOp3model_60_conv1d_145_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model_60/conv1d_145/BiasAdd/ReadVariableOpÜ
model_60/conv1d_145/BiasAddBiasAdd+model_60/conv1d_145/conv1d/Squeeze:output:02model_60/conv1d_145/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2
model_60/conv1d_145/BiasAdd
model_60/conv1d_145/ReluRelu$model_60/conv1d_145/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2
model_60/conv1d_145/Relu¡
)model_60/conv1d_144/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2+
)model_60/conv1d_144/conv1d/ExpandDims/dim
%model_60/conv1d_144/conv1d/ExpandDims
ExpandDims:model_60/embedding_60/embedding_lookup/Identity_1:output:02model_60/conv1d_144/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22'
%model_60/conv1d_144/conv1d/ExpandDimsô
6model_60/conv1d_144/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?model_60_conv1d_144_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype028
6model_60/conv1d_144/conv1d/ExpandDims_1/ReadVariableOp
+model_60/conv1d_144/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_60/conv1d_144/conv1d/ExpandDims_1/dim
'model_60/conv1d_144/conv1d/ExpandDims_1
ExpandDims>model_60/conv1d_144/conv1d/ExpandDims_1/ReadVariableOp:value:04model_60/conv1d_144/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2)
'model_60/conv1d_144/conv1d/ExpandDims_1
model_60/conv1d_144/conv1dConv2D.model_60/conv1d_144/conv1d/ExpandDims:output:00model_60/conv1d_144/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*
paddingVALID*
strides
2
model_60/conv1d_144/conv1dÎ
"model_60/conv1d_144/conv1d/SqueezeSqueeze#model_60/conv1d_144/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2$
"model_60/conv1d_144/conv1d/SqueezeÈ
*model_60/conv1d_144/BiasAdd/ReadVariableOpReadVariableOp3model_60_conv1d_144_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model_60/conv1d_144/BiasAdd/ReadVariableOpÜ
model_60/conv1d_144/BiasAddBiasAdd+model_60/conv1d_144/conv1d/Squeeze:output:02model_60/conv1d_144/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2
model_60/conv1d_144/BiasAdd
model_60/conv1d_144/ReluRelu$model_60/conv1d_144/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2
model_60/conv1d_144/Relu´
7model_60/global_max_pooling1d_144/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_60/global_max_pooling1d_144/Max/reduction_indicesñ
%model_60/global_max_pooling1d_144/MaxMax&model_60/conv1d_144/Relu:activations:0@model_60/global_max_pooling1d_144/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%model_60/global_max_pooling1d_144/Max´
7model_60/global_max_pooling1d_145/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_60/global_max_pooling1d_145/Max/reduction_indicesñ
%model_60/global_max_pooling1d_145/MaxMax&model_60/conv1d_145/Relu:activations:0@model_60/global_max_pooling1d_145/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%model_60/global_max_pooling1d_145/Max´
7model_60/global_max_pooling1d_146/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_60/global_max_pooling1d_146/Max/reduction_indicesñ
%model_60/global_max_pooling1d_146/MaxMax&model_60/conv1d_146/Relu:activations:0@model_60/global_max_pooling1d_146/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%model_60/global_max_pooling1d_146/Max
#model_60/concatenate_60/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_60/concatenate_60/concat/axisÆ
model_60/concatenate_60/concatConcatV2.model_60/global_max_pooling1d_144/Max:output:0.model_60/global_max_pooling1d_145/Max:output:0.model_60/global_max_pooling1d_146/Max:output:0,model_60/concatenate_60/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2 
model_60/concatenate_60/concat¤
model_60/dropout_60/IdentityIdentity'model_60/concatenate_60/concat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
model_60/dropout_60/IdentityÇ
(model_60/dense_135/MatMul/ReadVariableOpReadVariableOp1model_60_dense_135_matmul_readvariableop_resource*
_output_shapes
:	À*
dtype02*
(model_60/dense_135/MatMul/ReadVariableOpË
model_60/dense_135/MatMulMatMul%model_60/dropout_60/Identity:output:00model_60/dense_135/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_60/dense_135/MatMulÅ
)model_60/dense_135/BiasAdd/ReadVariableOpReadVariableOp2model_60_dense_135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_60/dense_135/BiasAdd/ReadVariableOpÍ
model_60/dense_135/BiasAddBiasAdd#model_60/dense_135/MatMul:product:01model_60/dense_135/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_60/dense_135/BiasAdd
model_60/dense_135/ReluRelu#model_60/dense_135/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_60/dense_135/ReluÆ
(model_60/dense_136/MatMul/ReadVariableOpReadVariableOp1model_60_dense_136_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(model_60/dense_136/MatMul/ReadVariableOpË
model_60/dense_136/MatMulMatMul%model_60/dense_135/Relu:activations:00model_60/dense_136/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_60/dense_136/MatMulÅ
)model_60/dense_136/BiasAdd/ReadVariableOpReadVariableOp2model_60_dense_136_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_60/dense_136/BiasAdd/ReadVariableOpÍ
model_60/dense_136/BiasAddBiasAdd#model_60/dense_136/MatMul:product:01model_60/dense_136/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_60/dense_136/BiasAdd
model_60/dense_136/SigmoidSigmoid#model_60/dense_136/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_60/dense_136/Sigmoidû
IdentityIdentitymodel_60/dense_136/Sigmoid:y:0+^model_60/conv1d_144/BiasAdd/ReadVariableOp7^model_60/conv1d_144/conv1d/ExpandDims_1/ReadVariableOp+^model_60/conv1d_145/BiasAdd/ReadVariableOp7^model_60/conv1d_145/conv1d/ExpandDims_1/ReadVariableOp+^model_60/conv1d_146/BiasAdd/ReadVariableOp7^model_60/conv1d_146/conv1d/ExpandDims_1/ReadVariableOp*^model_60/dense_135/BiasAdd/ReadVariableOp)^model_60/dense_135/MatMul/ReadVariableOp*^model_60/dense_136/BiasAdd/ReadVariableOp)^model_60/dense_136/MatMul/ReadVariableOp'^model_60/embedding_60/embedding_lookup*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::2X
*model_60/conv1d_144/BiasAdd/ReadVariableOp*model_60/conv1d_144/BiasAdd/ReadVariableOp2p
6model_60/conv1d_144/conv1d/ExpandDims_1/ReadVariableOp6model_60/conv1d_144/conv1d/ExpandDims_1/ReadVariableOp2X
*model_60/conv1d_145/BiasAdd/ReadVariableOp*model_60/conv1d_145/BiasAdd/ReadVariableOp2p
6model_60/conv1d_145/conv1d/ExpandDims_1/ReadVariableOp6model_60/conv1d_145/conv1d/ExpandDims_1/ReadVariableOp2X
*model_60/conv1d_146/BiasAdd/ReadVariableOp*model_60/conv1d_146/BiasAdd/ReadVariableOp2p
6model_60/conv1d_146/conv1d/ExpandDims_1/ReadVariableOp6model_60/conv1d_146/conv1d/ExpandDims_1/ReadVariableOp2V
)model_60/dense_135/BiasAdd/ReadVariableOp)model_60/dense_135/BiasAdd/ReadVariableOp2T
(model_60/dense_135/MatMul/ReadVariableOp(model_60/dense_135/MatMul/ReadVariableOp2V
)model_60/dense_136/BiasAdd/ReadVariableOp)model_60/dense_136/BiasAdd/ReadVariableOp2T
(model_60/dense_136/MatMul/ReadVariableOp(model_60/dense_136/MatMul/ReadVariableOp2P
&model_60/embedding_60/embedding_lookup&model_60/embedding_60/embedding_lookup:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
input_61
ó	
ß
F__inference_dense_135_layer_call_and_return_conditional_losses_1561016

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	À*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
õ
V
:__inference_global_max_pooling1d_145_layer_call_fn_1560819

inputs
identityß
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_145_layer_call_and_return_conditional_losses_15608132
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

*__inference_model_60_layer_call_fn_1561162
input_61
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinput_61unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_60_layer_call_and_return_conditional_losses_15611372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
input_61
Ü2
Ê
E__inference_model_60_layer_call_and_return_conditional_losses_1561097
input_61
embedding_60_1561063
conv1d_146_1561066
conv1d_146_1561068
conv1d_145_1561071
conv1d_145_1561073
conv1d_144_1561076
conv1d_144_1561078
dense_135_1561086
dense_135_1561088
dense_136_1561091
dense_136_1561093
identity¢"conv1d_144/StatefulPartitionedCall¢"conv1d_145/StatefulPartitionedCall¢"conv1d_146/StatefulPartitionedCall¢!dense_135/StatefulPartitionedCall¢!dense_136/StatefulPartitionedCall¢$embedding_60/StatefulPartitionedCall
$embedding_60/StatefulPartitionedCallStatefulPartitionedCallinput_61embedding_60_1561063*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_embedding_60_layer_call_and_return_conditional_losses_15608462&
$embedding_60/StatefulPartitionedCallÏ
"conv1d_146/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_146_1561066conv1d_146_1561068*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_146_layer_call_and_return_conditional_losses_15608742$
"conv1d_146/StatefulPartitionedCallÏ
"conv1d_145/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_145_1561071conv1d_145_1561073*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_145_layer_call_and_return_conditional_losses_15609062$
"conv1d_145/StatefulPartitionedCallÏ
"conv1d_144/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_144_1561076conv1d_144_1561078*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_144_layer_call_and_return_conditional_losses_15609382$
"conv1d_144/StatefulPartitionedCall­
(global_max_pooling1d_144/PartitionedCallPartitionedCall+conv1d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_144_layer_call_and_return_conditional_losses_15608002*
(global_max_pooling1d_144/PartitionedCall­
(global_max_pooling1d_145/PartitionedCallPartitionedCall+conv1d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_145_layer_call_and_return_conditional_losses_15608132*
(global_max_pooling1d_145/PartitionedCall­
(global_max_pooling1d_146/PartitionedCallPartitionedCall+conv1d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_146_layer_call_and_return_conditional_losses_15608262*
(global_max_pooling1d_146/PartitionedCallþ
concatenate_60/PartitionedCallPartitionedCall1global_max_pooling1d_144/PartitionedCall:output:01global_max_pooling1d_145/PartitionedCall:output:01global_max_pooling1d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_60_layer_call_and_return_conditional_losses_15609652 
concatenate_60/PartitionedCall
dropout_60/PartitionedCallPartitionedCall'concatenate_60/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_60_layer_call_and_return_conditional_losses_15609922
dropout_60/PartitionedCall¼
!dense_135/StatefulPartitionedCallStatefulPartitionedCall#dropout_60/PartitionedCall:output:0dense_135_1561086dense_135_1561088*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_135_layer_call_and_return_conditional_losses_15610162#
!dense_135/StatefulPartitionedCallÃ
!dense_136/StatefulPartitionedCallStatefulPartitionedCall*dense_135/StatefulPartitionedCall:output:0dense_136_1561091dense_136_1561093*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_136_layer_call_and_return_conditional_losses_15610432#
!dense_136/StatefulPartitionedCallÜ
IdentityIdentity*dense_136/StatefulPartitionedCall:output:0#^conv1d_144/StatefulPartitionedCall#^conv1d_145/StatefulPartitionedCall#^conv1d_146/StatefulPartitionedCall"^dense_135/StatefulPartitionedCall"^dense_136/StatefulPartitionedCall%^embedding_60/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::2H
"conv1d_144/StatefulPartitionedCall"conv1d_144/StatefulPartitionedCall2H
"conv1d_145/StatefulPartitionedCall"conv1d_145/StatefulPartitionedCall2H
"conv1d_146/StatefulPartitionedCall"conv1d_146/StatefulPartitionedCall2F
!dense_135/StatefulPartitionedCall!dense_135/StatefulPartitionedCall2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2L
$embedding_60/StatefulPartitionedCall$embedding_60/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
input_61
	

*__inference_model_60_layer_call_fn_1561435

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_60_layer_call_and_return_conditional_losses_15611372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

q
U__inference_global_max_pooling1d_145_layer_call_and_return_conditional_losses_1560813

inputs
identityp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

*__inference_model_60_layer_call_fn_1561226
input_61
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinput_61unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_60_layer_call_and_return_conditional_losses_15612012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
input_61

ú
G__inference_conv1d_145_layer_call_and_return_conditional_losses_1561520

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
 
_user_specified_nameinputs

ú
G__inference_conv1d_146_layer_call_and_return_conditional_losses_1561545

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
 
_user_specified_nameinputs
Á

K__inference_concatenate_60_layer_call_and_return_conditional_losses_1561562
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/2

H
,__inference_dropout_60_layer_call_fn_1561596

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_60_layer_call_and_return_conditional_losses_15609922
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

ú
G__inference_conv1d_146_layer_call_and_return_conditional_losses_1560874

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
 
_user_specified_nameinputs
	

*__inference_model_60_layer_call_fn_1561462

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_60_layer_call_and_return_conditional_losses_15612012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

ú
G__inference_conv1d_144_layer_call_and_return_conditional_losses_1561495

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
 
_user_specified_nameinputs
õ
V
:__inference_global_max_pooling1d_146_layer_call_fn_1560832

inputs
identityß
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_146_layer_call_and_return_conditional_losses_15608262
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
4
ï
E__inference_model_60_layer_call_and_return_conditional_losses_1561060
input_61
embedding_60_1560855
conv1d_146_1560885
conv1d_146_1560887
conv1d_145_1560917
conv1d_145_1560919
conv1d_144_1560949
conv1d_144_1560951
dense_135_1561027
dense_135_1561029
dense_136_1561054
dense_136_1561056
identity¢"conv1d_144/StatefulPartitionedCall¢"conv1d_145/StatefulPartitionedCall¢"conv1d_146/StatefulPartitionedCall¢!dense_135/StatefulPartitionedCall¢!dense_136/StatefulPartitionedCall¢"dropout_60/StatefulPartitionedCall¢$embedding_60/StatefulPartitionedCall
$embedding_60/StatefulPartitionedCallStatefulPartitionedCallinput_61embedding_60_1560855*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_embedding_60_layer_call_and_return_conditional_losses_15608462&
$embedding_60/StatefulPartitionedCallÏ
"conv1d_146/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_146_1560885conv1d_146_1560887*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_146_layer_call_and_return_conditional_losses_15608742$
"conv1d_146/StatefulPartitionedCallÏ
"conv1d_145/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_145_1560917conv1d_145_1560919*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_145_layer_call_and_return_conditional_losses_15609062$
"conv1d_145/StatefulPartitionedCallÏ
"conv1d_144/StatefulPartitionedCallStatefulPartitionedCall-embedding_60/StatefulPartitionedCall:output:0conv1d_144_1560949conv1d_144_1560951*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_144_layer_call_and_return_conditional_losses_15609382$
"conv1d_144/StatefulPartitionedCall­
(global_max_pooling1d_144/PartitionedCallPartitionedCall+conv1d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_144_layer_call_and_return_conditional_losses_15608002*
(global_max_pooling1d_144/PartitionedCall­
(global_max_pooling1d_145/PartitionedCallPartitionedCall+conv1d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_145_layer_call_and_return_conditional_losses_15608132*
(global_max_pooling1d_145/PartitionedCall­
(global_max_pooling1d_146/PartitionedCallPartitionedCall+conv1d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_146_layer_call_and_return_conditional_losses_15608262*
(global_max_pooling1d_146/PartitionedCallþ
concatenate_60/PartitionedCallPartitionedCall1global_max_pooling1d_144/PartitionedCall:output:01global_max_pooling1d_145/PartitionedCall:output:01global_max_pooling1d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_60_layer_call_and_return_conditional_losses_15609652 
concatenate_60/PartitionedCall
"dropout_60/StatefulPartitionedCallStatefulPartitionedCall'concatenate_60/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_60_layer_call_and_return_conditional_losses_15609872$
"dropout_60/StatefulPartitionedCallÄ
!dense_135/StatefulPartitionedCallStatefulPartitionedCall+dropout_60/StatefulPartitionedCall:output:0dense_135_1561027dense_135_1561029*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_135_layer_call_and_return_conditional_losses_15610162#
!dense_135/StatefulPartitionedCallÃ
!dense_136/StatefulPartitionedCallStatefulPartitionedCall*dense_135/StatefulPartitionedCall:output:0dense_136_1561054dense_136_1561056*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_136_layer_call_and_return_conditional_losses_15610432#
!dense_136/StatefulPartitionedCall
IdentityIdentity*dense_136/StatefulPartitionedCall:output:0#^conv1d_144/StatefulPartitionedCall#^conv1d_145/StatefulPartitionedCall#^conv1d_146/StatefulPartitionedCall"^dense_135/StatefulPartitionedCall"^dense_136/StatefulPartitionedCall#^dropout_60/StatefulPartitionedCall%^embedding_60/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::2H
"conv1d_144/StatefulPartitionedCall"conv1d_144/StatefulPartitionedCall2H
"conv1d_145/StatefulPartitionedCall"conv1d_145/StatefulPartitionedCall2H
"conv1d_146/StatefulPartitionedCall"conv1d_146/StatefulPartitionedCall2F
!dense_135/StatefulPartitionedCall!dense_135/StatefulPartitionedCall2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2H
"dropout_60/StatefulPartitionedCall"dropout_60/StatefulPartitionedCall2L
$embedding_60/StatefulPartitionedCall$embedding_60/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
input_61

ú
G__inference_conv1d_145_layer_call_and_return_conditional_losses_1560906

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
 
_user_specified_nameinputs

q
U__inference_global_max_pooling1d_144_layer_call_and_return_conditional_losses_1560800

inputs
identityp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
e
G__inference_dropout_60_layer_call_and_return_conditional_losses_1560992

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
å	

I__inference_embedding_60_layer_call_and_return_conditional_losses_1561472

inputs
embedding_lookup_1561466
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Castÿ
embedding_lookupResourceGatherembedding_lookup_1561466Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1561466*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
dtype02
embedding_lookupî
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1561466*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
½a
³
E__inference_model_60_layer_call_and_return_conditional_losses_1561408

inputs)
%embedding_60_embedding_lookup_1561343:
6conv1d_146_conv1d_expanddims_1_readvariableop_resource.
*conv1d_146_biasadd_readvariableop_resource:
6conv1d_145_conv1d_expanddims_1_readvariableop_resource.
*conv1d_145_biasadd_readvariableop_resource:
6conv1d_144_conv1d_expanddims_1_readvariableop_resource.
*conv1d_144_biasadd_readvariableop_resource,
(dense_135_matmul_readvariableop_resource-
)dense_135_biasadd_readvariableop_resource,
(dense_136_matmul_readvariableop_resource-
)dense_136_biasadd_readvariableop_resource
identity¢!conv1d_144/BiasAdd/ReadVariableOp¢-conv1d_144/conv1d/ExpandDims_1/ReadVariableOp¢!conv1d_145/BiasAdd/ReadVariableOp¢-conv1d_145/conv1d/ExpandDims_1/ReadVariableOp¢!conv1d_146/BiasAdd/ReadVariableOp¢-conv1d_146/conv1d/ExpandDims_1/ReadVariableOp¢ dense_135/BiasAdd/ReadVariableOp¢dense_135/MatMul/ReadVariableOp¢ dense_136/BiasAdd/ReadVariableOp¢dense_136/MatMul/ReadVariableOp¢embedding_60/embedding_lookupw
embedding_60/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
embedding_60/CastÀ
embedding_60/embedding_lookupResourceGather%embedding_60_embedding_lookup_1561343embedding_60/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_60/embedding_lookup/1561343*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
dtype02
embedding_60/embedding_lookup¢
&embedding_60/embedding_lookup/IdentityIdentity&embedding_60/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_60/embedding_lookup/1561343*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22(
&embedding_60/embedding_lookup/IdentityÇ
(embedding_60/embedding_lookup/Identity_1Identity/embedding_60/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22*
(embedding_60/embedding_lookup/Identity_1
 conv1d_146/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 conv1d_146/conv1d/ExpandDims/dimâ
conv1d_146/conv1d/ExpandDims
ExpandDims1embedding_60/embedding_lookup/Identity_1:output:0)conv1d_146/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d_146/conv1d/ExpandDimsÙ
-conv1d_146/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6conv1d_146_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02/
-conv1d_146/conv1d/ExpandDims_1/ReadVariableOp
"conv1d_146/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"conv1d_146/conv1d/ExpandDims_1/dimã
conv1d_146/conv1d/ExpandDims_1
ExpandDims5conv1d_146/conv1d/ExpandDims_1/ReadVariableOp:value:0+conv1d_146/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2 
conv1d_146/conv1d/ExpandDims_1ã
conv1d_146/conv1dConv2D%conv1d_146/conv1d/ExpandDims:output:0'conv1d_146/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*
paddingVALID*
strides
2
conv1d_146/conv1d³
conv1d_146/conv1d/SqueezeSqueezeconv1d_146/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_146/conv1d/Squeeze­
!conv1d_146/BiasAdd/ReadVariableOpReadVariableOp*conv1d_146_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv1d_146/BiasAdd/ReadVariableOp¸
conv1d_146/BiasAddBiasAdd"conv1d_146/conv1d/Squeeze:output:0)conv1d_146/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2
conv1d_146/BiasAdd}
conv1d_146/ReluReluconv1d_146/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2
conv1d_146/Relu
 conv1d_145/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 conv1d_145/conv1d/ExpandDims/dimâ
conv1d_145/conv1d/ExpandDims
ExpandDims1embedding_60/embedding_lookup/Identity_1:output:0)conv1d_145/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d_145/conv1d/ExpandDimsÙ
-conv1d_145/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6conv1d_145_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02/
-conv1d_145/conv1d/ExpandDims_1/ReadVariableOp
"conv1d_145/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"conv1d_145/conv1d/ExpandDims_1/dimã
conv1d_145/conv1d/ExpandDims_1
ExpandDims5conv1d_145/conv1d/ExpandDims_1/ReadVariableOp:value:0+conv1d_145/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2 
conv1d_145/conv1d/ExpandDims_1ã
conv1d_145/conv1dConv2D%conv1d_145/conv1d/ExpandDims:output:0'conv1d_145/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*
paddingVALID*
strides
2
conv1d_145/conv1d³
conv1d_145/conv1d/SqueezeSqueezeconv1d_145/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_145/conv1d/Squeeze­
!conv1d_145/BiasAdd/ReadVariableOpReadVariableOp*conv1d_145_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv1d_145/BiasAdd/ReadVariableOp¸
conv1d_145/BiasAddBiasAdd"conv1d_145/conv1d/Squeeze:output:0)conv1d_145/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2
conv1d_145/BiasAdd}
conv1d_145/ReluReluconv1d_145/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2
conv1d_145/Relu
 conv1d_144/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 conv1d_144/conv1d/ExpandDims/dimâ
conv1d_144/conv1d/ExpandDims
ExpandDims1embedding_60/embedding_lookup/Identity_1:output:0)conv1d_144/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d_144/conv1d/ExpandDimsÙ
-conv1d_144/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6conv1d_144_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02/
-conv1d_144/conv1d/ExpandDims_1/ReadVariableOp
"conv1d_144/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"conv1d_144/conv1d/ExpandDims_1/dimã
conv1d_144/conv1d/ExpandDims_1
ExpandDims5conv1d_144/conv1d/ExpandDims_1/ReadVariableOp:value:0+conv1d_144/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2 
conv1d_144/conv1d/ExpandDims_1ã
conv1d_144/conv1dConv2D%conv1d_144/conv1d/ExpandDims:output:0'conv1d_144/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*
paddingVALID*
strides
2
conv1d_144/conv1d³
conv1d_144/conv1d/SqueezeSqueezeconv1d_144/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_144/conv1d/Squeeze­
!conv1d_144/BiasAdd/ReadVariableOpReadVariableOp*conv1d_144_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv1d_144/BiasAdd/ReadVariableOp¸
conv1d_144/BiasAddBiasAdd"conv1d_144/conv1d/Squeeze:output:0)conv1d_144/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2
conv1d_144/BiasAdd}
conv1d_144/ReluReluconv1d_144/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2
conv1d_144/Relu¢
.global_max_pooling1d_144/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.global_max_pooling1d_144/Max/reduction_indicesÍ
global_max_pooling1d_144/MaxMaxconv1d_144/Relu:activations:07global_max_pooling1d_144/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
global_max_pooling1d_144/Max¢
.global_max_pooling1d_145/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.global_max_pooling1d_145/Max/reduction_indicesÍ
global_max_pooling1d_145/MaxMaxconv1d_145/Relu:activations:07global_max_pooling1d_145/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
global_max_pooling1d_145/Max¢
.global_max_pooling1d_146/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.global_max_pooling1d_146/Max/reduction_indicesÍ
global_max_pooling1d_146/MaxMaxconv1d_146/Relu:activations:07global_max_pooling1d_146/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
global_max_pooling1d_146/Maxz
concatenate_60/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_60/concat/axis
concatenate_60/concatConcatV2%global_max_pooling1d_144/Max:output:0%global_max_pooling1d_145/Max:output:0%global_max_pooling1d_146/Max:output:0#concatenate_60/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
concatenate_60/concat
dropout_60/IdentityIdentityconcatenate_60/concat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout_60/Identity¬
dense_135/MatMul/ReadVariableOpReadVariableOp(dense_135_matmul_readvariableop_resource*
_output_shapes
:	À*
dtype02!
dense_135/MatMul/ReadVariableOp§
dense_135/MatMulMatMuldropout_60/Identity:output:0'dense_135/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_135/MatMulª
 dense_135/BiasAdd/ReadVariableOpReadVariableOp)dense_135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_135/BiasAdd/ReadVariableOp©
dense_135/BiasAddBiasAdddense_135/MatMul:product:0(dense_135/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_135/BiasAddv
dense_135/ReluReludense_135/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_135/Relu«
dense_136/MatMul/ReadVariableOpReadVariableOp(dense_136_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_136/MatMul/ReadVariableOp§
dense_136/MatMulMatMuldense_135/Relu:activations:0'dense_136/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_136/MatMulª
 dense_136/BiasAdd/ReadVariableOpReadVariableOp)dense_136_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_136/BiasAdd/ReadVariableOp©
dense_136/BiasAddBiasAdddense_136/MatMul:product:0(dense_136/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_136/BiasAdd
dense_136/SigmoidSigmoiddense_136/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_136/Sigmoid
IdentityIdentitydense_136/Sigmoid:y:0"^conv1d_144/BiasAdd/ReadVariableOp.^conv1d_144/conv1d/ExpandDims_1/ReadVariableOp"^conv1d_145/BiasAdd/ReadVariableOp.^conv1d_145/conv1d/ExpandDims_1/ReadVariableOp"^conv1d_146/BiasAdd/ReadVariableOp.^conv1d_146/conv1d/ExpandDims_1/ReadVariableOp!^dense_135/BiasAdd/ReadVariableOp ^dense_135/MatMul/ReadVariableOp!^dense_136/BiasAdd/ReadVariableOp ^dense_136/MatMul/ReadVariableOp^embedding_60/embedding_lookup*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::2F
!conv1d_144/BiasAdd/ReadVariableOp!conv1d_144/BiasAdd/ReadVariableOp2^
-conv1d_144/conv1d/ExpandDims_1/ReadVariableOp-conv1d_144/conv1d/ExpandDims_1/ReadVariableOp2F
!conv1d_145/BiasAdd/ReadVariableOp!conv1d_145/BiasAdd/ReadVariableOp2^
-conv1d_145/conv1d/ExpandDims_1/ReadVariableOp-conv1d_145/conv1d/ExpandDims_1/ReadVariableOp2F
!conv1d_146/BiasAdd/ReadVariableOp!conv1d_146/BiasAdd/ReadVariableOp2^
-conv1d_146/conv1d/ExpandDims_1/ReadVariableOp-conv1d_146/conv1d/ExpandDims_1/ReadVariableOp2D
 dense_135/BiasAdd/ReadVariableOp dense_135/BiasAdd/ReadVariableOp2B
dense_135/MatMul/ReadVariableOpdense_135/MatMul/ReadVariableOp2D
 dense_136/BiasAdd/ReadVariableOp dense_136/BiasAdd/ReadVariableOp2B
dense_136/MatMul/ReadVariableOpdense_136/MatMul/ReadVariableOp2>
embedding_60/embedding_lookupembedding_60/embedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

q
U__inference_global_max_pooling1d_146_layer_call_and_return_conditional_losses_1560826

inputs
identityp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
e
G__inference_dropout_60_layer_call_and_return_conditional_losses_1561586

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

f
G__inference_dropout_60_layer_call_and_return_conditional_losses_1560987

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ñ
t
.__inference_embedding_60_layer_call_fn_1561479

inputs
unknown
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_embedding_60_layer_call_and_return_conditional_losses_15608462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
«
e
,__inference_dropout_60_layer_call_fn_1561591

inputs
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_60_layer_call_and_return_conditional_losses_15609872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
æ

+__inference_dense_135_layer_call_fn_1561616

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_135_layer_call_and_return_conditional_losses_15610162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
ö

,__inference_conv1d_145_layer_call_fn_1561529

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv1d_145_layer_call_and_return_conditional_losses_15609062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd2::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
 
_user_specified_nameinputs
Y
Ý
 __inference__traced_save_1561785
file_prefix6
2savev2_embedding_60_embeddings_read_readvariableop0
,savev2_conv1d_144_kernel_read_readvariableop.
*savev2_conv1d_144_bias_read_readvariableop0
,savev2_conv1d_145_kernel_read_readvariableop.
*savev2_conv1d_145_bias_read_readvariableop0
,savev2_conv1d_146_kernel_read_readvariableop.
*savev2_conv1d_146_bias_read_readvariableop/
+savev2_dense_135_kernel_read_readvariableop-
)savev2_dense_135_bias_read_readvariableop/
+savev2_dense_136_kernel_read_readvariableop-
)savev2_dense_136_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop=
9savev2_adam_embedding_60_embeddings_m_read_readvariableop7
3savev2_adam_conv1d_144_kernel_m_read_readvariableop5
1savev2_adam_conv1d_144_bias_m_read_readvariableop7
3savev2_adam_conv1d_145_kernel_m_read_readvariableop5
1savev2_adam_conv1d_145_bias_m_read_readvariableop7
3savev2_adam_conv1d_146_kernel_m_read_readvariableop5
1savev2_adam_conv1d_146_bias_m_read_readvariableop6
2savev2_adam_dense_135_kernel_m_read_readvariableop4
0savev2_adam_dense_135_bias_m_read_readvariableop6
2savev2_adam_dense_136_kernel_m_read_readvariableop4
0savev2_adam_dense_136_bias_m_read_readvariableop=
9savev2_adam_embedding_60_embeddings_v_read_readvariableop7
3savev2_adam_conv1d_144_kernel_v_read_readvariableop5
1savev2_adam_conv1d_144_bias_v_read_readvariableop7
3savev2_adam_conv1d_145_kernel_v_read_readvariableop5
1savev2_adam_conv1d_145_bias_v_read_readvariableop7
3savev2_adam_conv1d_146_kernel_v_read_readvariableop5
1savev2_adam_conv1d_146_bias_v_read_readvariableop6
2savev2_adam_dense_135_kernel_v_read_readvariableop4
0savev2_adam_dense_135_bias_v_read_readvariableop6
2savev2_adam_dense_136_kernel_v_read_readvariableop4
0savev2_adam_dense_136_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameì
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*þ
valueôBñ+B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÞ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¬
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_embedding_60_embeddings_read_readvariableop,savev2_conv1d_144_kernel_read_readvariableop*savev2_conv1d_144_bias_read_readvariableop,savev2_conv1d_145_kernel_read_readvariableop*savev2_conv1d_145_bias_read_readvariableop,savev2_conv1d_146_kernel_read_readvariableop*savev2_conv1d_146_bias_read_readvariableop+savev2_dense_135_kernel_read_readvariableop)savev2_dense_135_bias_read_readvariableop+savev2_dense_136_kernel_read_readvariableop)savev2_dense_136_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop9savev2_adam_embedding_60_embeddings_m_read_readvariableop3savev2_adam_conv1d_144_kernel_m_read_readvariableop1savev2_adam_conv1d_144_bias_m_read_readvariableop3savev2_adam_conv1d_145_kernel_m_read_readvariableop1savev2_adam_conv1d_145_bias_m_read_readvariableop3savev2_adam_conv1d_146_kernel_m_read_readvariableop1savev2_adam_conv1d_146_bias_m_read_readvariableop2savev2_adam_dense_135_kernel_m_read_readvariableop0savev2_adam_dense_135_bias_m_read_readvariableop2savev2_adam_dense_136_kernel_m_read_readvariableop0savev2_adam_dense_136_bias_m_read_readvariableop9savev2_adam_embedding_60_embeddings_v_read_readvariableop3savev2_adam_conv1d_144_kernel_v_read_readvariableop1savev2_adam_conv1d_144_bias_v_read_readvariableop3savev2_adam_conv1d_145_kernel_v_read_readvariableop1savev2_adam_conv1d_145_bias_v_read_readvariableop3savev2_adam_conv1d_146_kernel_v_read_readvariableop1savev2_adam_conv1d_146_bias_v_read_readvariableop2savev2_adam_dense_135_kernel_v_read_readvariableop0savev2_adam_dense_135_bias_v_read_readvariableop2savev2_adam_dense_136_kernel_v_read_readvariableop0savev2_adam_dense_136_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ã
_input_shapesÑ
Î: :	@2:2@:@:2@:@:2@:@:	À:::: : : : : : : : : :	@2:2@:@:2@:@:2@:@:	À::::	@2:2@:@:2@:@:2@:@:	À:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	@2:($
"
_output_shapes
:2@: 

_output_shapes
:@:($
"
_output_shapes
:2@: 

_output_shapes
:@:($
"
_output_shapes
:2@: 

_output_shapes
:@:%!

_output_shapes
:	À: 	

_output_shapes
::$
 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	@2:($
"
_output_shapes
:2@: 

_output_shapes
:@:($
"
_output_shapes
:2@: 

_output_shapes
:@:($
"
_output_shapes
:2@: 

_output_shapes
:@:%!

_output_shapes
:	À: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::% !

_output_shapes
:	@2:(!$
"
_output_shapes
:2@: "

_output_shapes
:@:(#$
"
_output_shapes
:2@: $

_output_shapes
:@:(%$
"
_output_shapes
:2@: &

_output_shapes
:@:%'!

_output_shapes
:	À: (

_output_shapes
::$) 

_output_shapes

:: *

_output_shapes
::+

_output_shapes
: 
²

#__inference__traced_restore_1561921
file_prefix,
(assignvariableop_embedding_60_embeddings(
$assignvariableop_1_conv1d_144_kernel&
"assignvariableop_2_conv1d_144_bias(
$assignvariableop_3_conv1d_145_kernel&
"assignvariableop_4_conv1d_145_bias(
$assignvariableop_5_conv1d_146_kernel&
"assignvariableop_6_conv1d_146_bias'
#assignvariableop_7_dense_135_kernel%
!assignvariableop_8_dense_135_bias'
#assignvariableop_9_dense_136_kernel&
"assignvariableop_10_dense_136_bias!
assignvariableop_11_adam_iter#
assignvariableop_12_adam_beta_1#
assignvariableop_13_adam_beta_2"
assignvariableop_14_adam_decay*
&assignvariableop_15_adam_learning_rate
assignvariableop_16_total
assignvariableop_17_count
assignvariableop_18_total_1
assignvariableop_19_count_16
2assignvariableop_20_adam_embedding_60_embeddings_m0
,assignvariableop_21_adam_conv1d_144_kernel_m.
*assignvariableop_22_adam_conv1d_144_bias_m0
,assignvariableop_23_adam_conv1d_145_kernel_m.
*assignvariableop_24_adam_conv1d_145_bias_m0
,assignvariableop_25_adam_conv1d_146_kernel_m.
*assignvariableop_26_adam_conv1d_146_bias_m/
+assignvariableop_27_adam_dense_135_kernel_m-
)assignvariableop_28_adam_dense_135_bias_m/
+assignvariableop_29_adam_dense_136_kernel_m-
)assignvariableop_30_adam_dense_136_bias_m6
2assignvariableop_31_adam_embedding_60_embeddings_v0
,assignvariableop_32_adam_conv1d_144_kernel_v.
*assignvariableop_33_adam_conv1d_144_bias_v0
,assignvariableop_34_adam_conv1d_145_kernel_v.
*assignvariableop_35_adam_conv1d_145_bias_v0
,assignvariableop_36_adam_conv1d_146_kernel_v.
*assignvariableop_37_adam_conv1d_146_bias_v/
+assignvariableop_38_adam_dense_135_kernel_v-
)assignvariableop_39_adam_dense_135_bias_v/
+assignvariableop_40_adam_dense_136_kernel_v-
)assignvariableop_41_adam_dense_136_bias_v
identity_43¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ò
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*þ
valueôBñ+B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesä
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapes¯
¬:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity§
AssignVariableOpAssignVariableOp(assignvariableop_embedding_60_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1©
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv1d_144_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_144_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3©
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv1d_145_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_145_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5©
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv1d_146_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_146_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¨
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_135_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¦
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_135_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¨
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_136_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ª
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_136_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11¥
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12§
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¦
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15®
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¡
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¡
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18£
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19£
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20º
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_embedding_60_embeddings_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21´
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_conv1d_144_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22²
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_conv1d_144_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23´
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_conv1d_145_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_conv1d_145_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25´
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_conv1d_146_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26²
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_conv1d_146_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_135_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_135_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_136_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_136_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31º
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_embedding_60_embeddings_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32´
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_conv1d_144_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33²
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_144_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34´
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_conv1d_145_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv1d_145_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36´
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_conv1d_146_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv1d_146_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38³
AssignVariableOp_38AssignVariableOp+assignvariableop_38_adam_dense_135_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39±
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_135_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40³
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_dense_136_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41±
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_136_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_419
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpú
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_42í
Identity_43IdentityIdentity_42:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_43"#
identity_43Identity_43:output:0*¿
_input_shapes­
ª: ::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ó	
ß
F__inference_dense_135_layer_call_and_return_conditional_losses_1561607

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	À*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
õ
V
:__inference_global_max_pooling1d_144_layer_call_fn_1560806

inputs
identityß
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_max_pooling1d_144_layer_call_and_return_conditional_losses_15608002
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
k
³
E__inference_model_60_layer_call_and_return_conditional_losses_1561339

inputs)
%embedding_60_embedding_lookup_1561267:
6conv1d_146_conv1d_expanddims_1_readvariableop_resource.
*conv1d_146_biasadd_readvariableop_resource:
6conv1d_145_conv1d_expanddims_1_readvariableop_resource.
*conv1d_145_biasadd_readvariableop_resource:
6conv1d_144_conv1d_expanddims_1_readvariableop_resource.
*conv1d_144_biasadd_readvariableop_resource,
(dense_135_matmul_readvariableop_resource-
)dense_135_biasadd_readvariableop_resource,
(dense_136_matmul_readvariableop_resource-
)dense_136_biasadd_readvariableop_resource
identity¢!conv1d_144/BiasAdd/ReadVariableOp¢-conv1d_144/conv1d/ExpandDims_1/ReadVariableOp¢!conv1d_145/BiasAdd/ReadVariableOp¢-conv1d_145/conv1d/ExpandDims_1/ReadVariableOp¢!conv1d_146/BiasAdd/ReadVariableOp¢-conv1d_146/conv1d/ExpandDims_1/ReadVariableOp¢ dense_135/BiasAdd/ReadVariableOp¢dense_135/MatMul/ReadVariableOp¢ dense_136/BiasAdd/ReadVariableOp¢dense_136/MatMul/ReadVariableOp¢embedding_60/embedding_lookupw
embedding_60/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
embedding_60/CastÀ
embedding_60/embedding_lookupResourceGather%embedding_60_embedding_lookup_1561267embedding_60/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_60/embedding_lookup/1561267*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
dtype02
embedding_60/embedding_lookup¢
&embedding_60/embedding_lookup/IdentityIdentity&embedding_60/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_60/embedding_lookup/1561267*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22(
&embedding_60/embedding_lookup/IdentityÇ
(embedding_60/embedding_lookup/Identity_1Identity/embedding_60/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22*
(embedding_60/embedding_lookup/Identity_1
 conv1d_146/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 conv1d_146/conv1d/ExpandDims/dimâ
conv1d_146/conv1d/ExpandDims
ExpandDims1embedding_60/embedding_lookup/Identity_1:output:0)conv1d_146/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d_146/conv1d/ExpandDimsÙ
-conv1d_146/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6conv1d_146_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02/
-conv1d_146/conv1d/ExpandDims_1/ReadVariableOp
"conv1d_146/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"conv1d_146/conv1d/ExpandDims_1/dimã
conv1d_146/conv1d/ExpandDims_1
ExpandDims5conv1d_146/conv1d/ExpandDims_1/ReadVariableOp:value:0+conv1d_146/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2 
conv1d_146/conv1d/ExpandDims_1ã
conv1d_146/conv1dConv2D%conv1d_146/conv1d/ExpandDims:output:0'conv1d_146/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*
paddingVALID*
strides
2
conv1d_146/conv1d³
conv1d_146/conv1d/SqueezeSqueezeconv1d_146/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_146/conv1d/Squeeze­
!conv1d_146/BiasAdd/ReadVariableOpReadVariableOp*conv1d_146_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv1d_146/BiasAdd/ReadVariableOp¸
conv1d_146/BiasAddBiasAdd"conv1d_146/conv1d/Squeeze:output:0)conv1d_146/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2
conv1d_146/BiasAdd}
conv1d_146/ReluReluconv1d_146/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿU@2
conv1d_146/Relu
 conv1d_145/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 conv1d_145/conv1d/ExpandDims/dimâ
conv1d_145/conv1d/ExpandDims
ExpandDims1embedding_60/embedding_lookup/Identity_1:output:0)conv1d_145/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d_145/conv1d/ExpandDimsÙ
-conv1d_145/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6conv1d_145_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02/
-conv1d_145/conv1d/ExpandDims_1/ReadVariableOp
"conv1d_145/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"conv1d_145/conv1d/ExpandDims_1/dimã
conv1d_145/conv1d/ExpandDims_1
ExpandDims5conv1d_145/conv1d/ExpandDims_1/ReadVariableOp:value:0+conv1d_145/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2 
conv1d_145/conv1d/ExpandDims_1ã
conv1d_145/conv1dConv2D%conv1d_145/conv1d/ExpandDims:output:0'conv1d_145/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*
paddingVALID*
strides
2
conv1d_145/conv1d³
conv1d_145/conv1d/SqueezeSqueezeconv1d_145/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_145/conv1d/Squeeze­
!conv1d_145/BiasAdd/ReadVariableOpReadVariableOp*conv1d_145_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv1d_145/BiasAdd/ReadVariableOp¸
conv1d_145/BiasAddBiasAdd"conv1d_145/conv1d/Squeeze:output:0)conv1d_145/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2
conv1d_145/BiasAdd}
conv1d_145/ReluReluconv1d_145/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]@2
conv1d_145/Relu
 conv1d_144/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 conv1d_144/conv1d/ExpandDims/dimâ
conv1d_144/conv1d/ExpandDims
ExpandDims1embedding_60/embedding_lookup/Identity_1:output:0)conv1d_144/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d_144/conv1d/ExpandDimsÙ
-conv1d_144/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6conv1d_144_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02/
-conv1d_144/conv1d/ExpandDims_1/ReadVariableOp
"conv1d_144/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"conv1d_144/conv1d/ExpandDims_1/dimã
conv1d_144/conv1d/ExpandDims_1
ExpandDims5conv1d_144/conv1d/ExpandDims_1/ReadVariableOp:value:0+conv1d_144/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2 
conv1d_144/conv1d/ExpandDims_1ã
conv1d_144/conv1dConv2D%conv1d_144/conv1d/ExpandDims:output:0'conv1d_144/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*
paddingVALID*
strides
2
conv1d_144/conv1d³
conv1d_144/conv1d/SqueezeSqueezeconv1d_144/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_144/conv1d/Squeeze­
!conv1d_144/BiasAdd/ReadVariableOpReadVariableOp*conv1d_144_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv1d_144/BiasAdd/ReadVariableOp¸
conv1d_144/BiasAddBiasAdd"conv1d_144/conv1d/Squeeze:output:0)conv1d_144/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2
conv1d_144/BiasAdd}
conv1d_144/ReluReluconv1d_144/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2
conv1d_144/Relu¢
.global_max_pooling1d_144/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.global_max_pooling1d_144/Max/reduction_indicesÍ
global_max_pooling1d_144/MaxMaxconv1d_144/Relu:activations:07global_max_pooling1d_144/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
global_max_pooling1d_144/Max¢
.global_max_pooling1d_145/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.global_max_pooling1d_145/Max/reduction_indicesÍ
global_max_pooling1d_145/MaxMaxconv1d_145/Relu:activations:07global_max_pooling1d_145/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
global_max_pooling1d_145/Max¢
.global_max_pooling1d_146/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.global_max_pooling1d_146/Max/reduction_indicesÍ
global_max_pooling1d_146/MaxMaxconv1d_146/Relu:activations:07global_max_pooling1d_146/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
global_max_pooling1d_146/Maxz
concatenate_60/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_60/concat/axis
concatenate_60/concatConcatV2%global_max_pooling1d_144/Max:output:0%global_max_pooling1d_145/Max:output:0%global_max_pooling1d_146/Max:output:0#concatenate_60/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
concatenate_60/concaty
dropout_60/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_60/dropout/Const­
dropout_60/dropout/MulMulconcatenate_60/concat:output:0!dropout_60/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout_60/dropout/Mul
dropout_60/dropout/ShapeShapeconcatenate_60/concat:output:0*
T0*
_output_shapes
:2
dropout_60/dropout/ShapeÖ
/dropout_60/dropout/random_uniform/RandomUniformRandomUniform!dropout_60/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
dtype021
/dropout_60/dropout/random_uniform/RandomUniform
!dropout_60/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!dropout_60/dropout/GreaterEqual/yë
dropout_60/dropout/GreaterEqualGreaterEqual8dropout_60/dropout/random_uniform/RandomUniform:output:0*dropout_60/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2!
dropout_60/dropout/GreaterEqual¡
dropout_60/dropout/CastCast#dropout_60/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout_60/dropout/Cast§
dropout_60/dropout/Mul_1Muldropout_60/dropout/Mul:z:0dropout_60/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout_60/dropout/Mul_1¬
dense_135/MatMul/ReadVariableOpReadVariableOp(dense_135_matmul_readvariableop_resource*
_output_shapes
:	À*
dtype02!
dense_135/MatMul/ReadVariableOp§
dense_135/MatMulMatMuldropout_60/dropout/Mul_1:z:0'dense_135/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_135/MatMulª
 dense_135/BiasAdd/ReadVariableOpReadVariableOp)dense_135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_135/BiasAdd/ReadVariableOp©
dense_135/BiasAddBiasAdddense_135/MatMul:product:0(dense_135/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_135/BiasAddv
dense_135/ReluReludense_135/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_135/Relu«
dense_136/MatMul/ReadVariableOpReadVariableOp(dense_136_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_136/MatMul/ReadVariableOp§
dense_136/MatMulMatMuldense_135/Relu:activations:0'dense_136/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_136/MatMulª
 dense_136/BiasAdd/ReadVariableOpReadVariableOp)dense_136_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_136/BiasAdd/ReadVariableOp©
dense_136/BiasAddBiasAdddense_136/MatMul:product:0(dense_136/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_136/BiasAdd
dense_136/SigmoidSigmoiddense_136/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_136/Sigmoid
IdentityIdentitydense_136/Sigmoid:y:0"^conv1d_144/BiasAdd/ReadVariableOp.^conv1d_144/conv1d/ExpandDims_1/ReadVariableOp"^conv1d_145/BiasAdd/ReadVariableOp.^conv1d_145/conv1d/ExpandDims_1/ReadVariableOp"^conv1d_146/BiasAdd/ReadVariableOp.^conv1d_146/conv1d/ExpandDims_1/ReadVariableOp!^dense_135/BiasAdd/ReadVariableOp ^dense_135/MatMul/ReadVariableOp!^dense_136/BiasAdd/ReadVariableOp ^dense_136/MatMul/ReadVariableOp^embedding_60/embedding_lookup*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:::::::::::2F
!conv1d_144/BiasAdd/ReadVariableOp!conv1d_144/BiasAdd/ReadVariableOp2^
-conv1d_144/conv1d/ExpandDims_1/ReadVariableOp-conv1d_144/conv1d/ExpandDims_1/ReadVariableOp2F
!conv1d_145/BiasAdd/ReadVariableOp!conv1d_145/BiasAdd/ReadVariableOp2^
-conv1d_145/conv1d/ExpandDims_1/ReadVariableOp-conv1d_145/conv1d/ExpandDims_1/ReadVariableOp2F
!conv1d_146/BiasAdd/ReadVariableOp!conv1d_146/BiasAdd/ReadVariableOp2^
-conv1d_146/conv1d/ExpandDims_1/ReadVariableOp-conv1d_146/conv1d/ExpandDims_1/ReadVariableOp2D
 dense_135/BiasAdd/ReadVariableOp dense_135/BiasAdd/ReadVariableOp2B
dense_135/MatMul/ReadVariableOpdense_135/MatMul/ReadVariableOp2D
 dense_136/BiasAdd/ReadVariableOp dense_136/BiasAdd/ReadVariableOp2B
dense_136/MatMul/ReadVariableOpdense_136/MatMul/ReadVariableOp2>
embedding_60/embedding_lookupembedding_60/embedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
©
j
0__inference_concatenate_60_layer_call_fn_1561569
inputs_0
inputs_1
inputs_2
identityå
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_60_layer_call_and_return_conditional_losses_15609652
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/2
ò	
ß
F__inference_dense_136_layer_call_and_return_conditional_losses_1561627

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
G__inference_dropout_60_layer_call_and_return_conditional_losses_1561581

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
ä

+__inference_dense_136_layer_call_fn_1561636

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_136_layer_call_and_return_conditional_losses_15610432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ú
G__inference_conv1d_144_layer_call_and_return_conditional_losses_1560938

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2@2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
 
_user_specified_nameinputs
·

K__inference_concatenate_60_layer_call_and_return_conditional_losses_1560965

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*®
serving_default
=
input_611
serving_default_input_61:0ÿÿÿÿÿÿÿÿÿd=
	dense_1360
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:êß
¨_
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
¬_default_save_signature
+­&call_and_return_all_conditional_losses
®__call__" [
_tf_keras_network[{"class_name": "Functional", "name": "model_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_60", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_61"}, "name": "input_61", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_60", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 8196, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_60", "inbound_nodes": [[["input_61", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_144", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_144", "inbound_nodes": [[["embedding_60", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_145", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_145", "inbound_nodes": [[["embedding_60", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_146", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_146", "inbound_nodes": [[["embedding_60", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "global_max_pooling1d_144", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling1d_144", "inbound_nodes": [[["conv1d_144", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "global_max_pooling1d_145", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling1d_145", "inbound_nodes": [[["conv1d_145", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "global_max_pooling1d_146", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling1d_146", "inbound_nodes": [[["conv1d_146", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_60", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concatenate_60", "inbound_nodes": [[["global_max_pooling1d_144", 0, 0, {}], ["global_max_pooling1d_145", 0, 0, {}], ["global_max_pooling1d_146", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_60", "inbound_nodes": [[["concatenate_60", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_135", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_135", "inbound_nodes": [[["dropout_60", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_136", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_136", "inbound_nodes": [[["dense_135", 0, 0, {}]]]}], "input_layers": [["input_61", 0, 0]], "output_layers": [["dense_136", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_60", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_61"}, "name": "input_61", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_60", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 8196, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_60", "inbound_nodes": [[["input_61", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_144", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_144", "inbound_nodes": [[["embedding_60", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_145", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_145", "inbound_nodes": [[["embedding_60", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_146", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_146", "inbound_nodes": [[["embedding_60", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "global_max_pooling1d_144", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling1d_144", "inbound_nodes": [[["conv1d_144", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "global_max_pooling1d_145", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling1d_145", "inbound_nodes": [[["conv1d_145", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "global_max_pooling1d_146", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling1d_146", "inbound_nodes": [[["conv1d_146", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_60", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concatenate_60", "inbound_nodes": [[["global_max_pooling1d_144", 0, 0, {}], ["global_max_pooling1d_145", 0, 0, {}], ["global_max_pooling1d_146", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_60", "inbound_nodes": [[["concatenate_60", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_135", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_135", "inbound_nodes": [[["dropout_60", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_136", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_136", "inbound_nodes": [[["dense_135", 0, 0, {}]]]}], "input_layers": [["input_61", 0, 0]], "output_layers": [["dense_136", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ï"ì
_tf_keras_input_layerÌ{"class_name": "InputLayer", "name": "input_61", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_61"}}
³

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
+¯&call_and_return_all_conditional_losses
°__call__"
_tf_keras_layerø{"class_name": "Embedding", "name": "embedding_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_60", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 8196, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
í	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+±&call_and_return_all_conditional_losses
²__call__"Æ
_tf_keras_layer¬{"class_name": "Conv1D", "name": "conv1d_144", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_144", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 50]}}
í	

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
+³&call_and_return_all_conditional_losses
´__call__"Æ
_tf_keras_layer¬{"class_name": "Conv1D", "name": "conv1d_145", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_145", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 50]}}
î	

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+µ&call_and_return_all_conditional_losses
¶__call__"Ç
_tf_keras_layer­{"class_name": "Conv1D", "name": "conv1d_146", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_146", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 50]}}

*	variables
+trainable_variables
,regularization_losses
-	keras_api
+·&call_and_return_all_conditional_losses
¸__call__"
_tf_keras_layeræ{"class_name": "GlobalMaxPooling1D", "name": "global_max_pooling1d_144", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_max_pooling1d_144", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

.	variables
/trainable_variables
0regularization_losses
1	keras_api
+¹&call_and_return_all_conditional_losses
º__call__"
_tf_keras_layeræ{"class_name": "GlobalMaxPooling1D", "name": "global_max_pooling1d_145", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_max_pooling1d_145", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

2	variables
3trainable_variables
4regularization_losses
5	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"
_tf_keras_layeræ{"class_name": "GlobalMaxPooling1D", "name": "global_max_pooling1d_146", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_max_pooling1d_146", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

6	variables
7trainable_variables
8regularization_losses
9	keras_api
+½&call_and_return_all_conditional_losses
¾__call__"ó
_tf_keras_layerÙ{"class_name": "Concatenate", "name": "concatenate_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_60", "trainable": true, "dtype": "float32", "axis": 1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64]}, {"class_name": "TensorShape", "items": [null, 64]}, {"class_name": "TensorShape", "items": [null, 64]}]}
é
:	variables
;trainable_variables
<regularization_losses
=	keras_api
+¿&call_and_return_all_conditional_losses
À__call__"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
÷

>kernel
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
+Á&call_and_return_all_conditional_losses
Â__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_135", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_135", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192]}}
ö

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
+Ã&call_and_return_all_conditional_losses
Ä__call__"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_136", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_136", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
¯
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratemmmmm$m%m>m?mDmEm v¡v¢v£v¤v¥$v¦%v§>v¨?v©DvªEv«"
	optimizer
n
0
1
2
3
4
$5
%6
>7
?8
D9
E10"
trackable_list_wrapper
n
0
1
2
3
4
$5
%6
>7
?8
D9
E10"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
	variables

Olayers
trainable_variables
Player_regularization_losses
Qlayer_metrics
Rnon_trainable_variables
Smetrics
regularization_losses
®__call__
¬_default_save_signature
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
-
Åserving_default"
signature_map
*:(	@22embedding_60/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
°
	variables

Tlayers
trainable_variables
Ulayer_metrics
Vlayer_regularization_losses
Wnon_trainable_variables
Xmetrics
regularization_losses
°__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
':%2@2conv1d_144/kernel
:@2conv1d_144/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
	variables

Ylayers
trainable_variables
Zlayer_metrics
[layer_regularization_losses
\non_trainable_variables
]metrics
regularization_losses
²__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
':%2@2conv1d_145/kernel
:@2conv1d_145/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
 	variables

^layers
!trainable_variables
_layer_metrics
`layer_regularization_losses
anon_trainable_variables
bmetrics
"regularization_losses
´__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
':%2@2conv1d_146/kernel
:@2conv1d_146/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
&	variables

clayers
'trainable_variables
dlayer_metrics
elayer_regularization_losses
fnon_trainable_variables
gmetrics
(regularization_losses
¶__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
*	variables

hlayers
+trainable_variables
ilayer_metrics
jlayer_regularization_losses
knon_trainable_variables
lmetrics
,regularization_losses
¸__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
.	variables

mlayers
/trainable_variables
nlayer_metrics
olayer_regularization_losses
pnon_trainable_variables
qmetrics
0regularization_losses
º__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
2	variables

rlayers
3trainable_variables
slayer_metrics
tlayer_regularization_losses
unon_trainable_variables
vmetrics
4regularization_losses
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
6	variables

wlayers
7trainable_variables
xlayer_metrics
ylayer_regularization_losses
znon_trainable_variables
{metrics
8regularization_losses
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
±
:	variables

|layers
;trainable_variables
}layer_metrics
~layer_regularization_losses
non_trainable_variables
metrics
<regularization_losses
À__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
#:!	À2dense_135/kernel
:2dense_135/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
@	variables
layers
Atrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
metrics
Bregularization_losses
Â__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
": 2dense_136/kernel
:2dense_136/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
F	variables
layers
Gtrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
metrics
Hregularization_losses
Ä__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ÿ

total

count

_fn_kwargs
	variables
	keras_api"³
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
/:-	@22Adam/embedding_60/embeddings/m
,:*2@2Adam/conv1d_144/kernel/m
": @2Adam/conv1d_144/bias/m
,:*2@2Adam/conv1d_145/kernel/m
": @2Adam/conv1d_145/bias/m
,:*2@2Adam/conv1d_146/kernel/m
": @2Adam/conv1d_146/bias/m
(:&	À2Adam/dense_135/kernel/m
!:2Adam/dense_135/bias/m
':%2Adam/dense_136/kernel/m
!:2Adam/dense_136/bias/m
/:-	@22Adam/embedding_60/embeddings/v
,:*2@2Adam/conv1d_144/kernel/v
": @2Adam/conv1d_144/bias/v
,:*2@2Adam/conv1d_145/kernel/v
": @2Adam/conv1d_145/bias/v
,:*2@2Adam/conv1d_146/kernel/v
": @2Adam/conv1d_146/bias/v
(:&	À2Adam/dense_135/kernel/v
!:2Adam/dense_135/bias/v
':%2Adam/dense_136/kernel/v
!:2Adam/dense_136/bias/v
á2Þ
"__inference__wrapped_model_1560793·
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *'¢$
"
input_61ÿÿÿÿÿÿÿÿÿd
â2ß
E__inference_model_60_layer_call_and_return_conditional_losses_1561060
E__inference_model_60_layer_call_and_return_conditional_losses_1561339
E__inference_model_60_layer_call_and_return_conditional_losses_1561097
E__inference_model_60_layer_call_and_return_conditional_losses_1561408À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
*__inference_model_60_layer_call_fn_1561435
*__inference_model_60_layer_call_fn_1561462
*__inference_model_60_layer_call_fn_1561226
*__inference_model_60_layer_call_fn_1561162À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ó2ð
I__inference_embedding_60_layer_call_and_return_conditional_losses_1561472¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_embedding_60_layer_call_fn_1561479¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_conv1d_144_layer_call_and_return_conditional_losses_1561495¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_conv1d_144_layer_call_fn_1561504¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_conv1d_145_layer_call_and_return_conditional_losses_1561520¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_conv1d_145_layer_call_fn_1561529¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_conv1d_146_layer_call_and_return_conditional_losses_1561545¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_conv1d_146_layer_call_fn_1561554¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
°2­
U__inference_global_max_pooling1d_144_layer_call_and_return_conditional_losses_1560800Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
:__inference_global_max_pooling1d_144_layer_call_fn_1560806Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°2­
U__inference_global_max_pooling1d_145_layer_call_and_return_conditional_losses_1560813Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
:__inference_global_max_pooling1d_145_layer_call_fn_1560819Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°2­
U__inference_global_max_pooling1d_146_layer_call_and_return_conditional_losses_1560826Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
:__inference_global_max_pooling1d_146_layer_call_fn_1560832Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
õ2ò
K__inference_concatenate_60_layer_call_and_return_conditional_losses_1561562¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_concatenate_60_layer_call_fn_1561569¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ì2É
G__inference_dropout_60_layer_call_and_return_conditional_losses_1561586
G__inference_dropout_60_layer_call_and_return_conditional_losses_1561581´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
,__inference_dropout_60_layer_call_fn_1561596
,__inference_dropout_60_layer_call_fn_1561591´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_dense_135_layer_call_and_return_conditional_losses_1561607¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_135_layer_call_fn_1561616¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_136_layer_call_and_return_conditional_losses_1561627¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_136_layer_call_fn_1561636¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
%__inference_signature_wrapper_1561263input_61"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"__inference__wrapped_model_1560793w$%>?DE1¢.
'¢$
"
input_61ÿÿÿÿÿÿÿÿÿd
ª "5ª2
0
	dense_136# 
	dense_136ÿÿÿÿÿÿÿÿÿø
K__inference_concatenate_60_layer_call_and_return_conditional_losses_1561562¨~¢{
t¢q
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@
"
inputs/2ÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 Ð
0__inference_concatenate_60_layer_call_fn_1561569~¢{
t¢q
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@
"
inputs/2ÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿÀ¯
G__inference_conv1d_144_layer_call_and_return_conditional_losses_1561495d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿd2
ª ")¢&

0ÿÿÿÿÿÿÿÿÿa@
 
,__inference_conv1d_144_layer_call_fn_1561504W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿd2
ª "ÿÿÿÿÿÿÿÿÿa@¯
G__inference_conv1d_145_layer_call_and_return_conditional_losses_1561520d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿd2
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ]@
 
,__inference_conv1d_145_layer_call_fn_1561529W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿd2
ª "ÿÿÿÿÿÿÿÿÿ]@¯
G__inference_conv1d_146_layer_call_and_return_conditional_losses_1561545d$%3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿd2
ª ")¢&

0ÿÿÿÿÿÿÿÿÿU@
 
,__inference_conv1d_146_layer_call_fn_1561554W$%3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿd2
ª "ÿÿÿÿÿÿÿÿÿU@§
F__inference_dense_135_layer_call_and_return_conditional_losses_1561607]>?0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_135_layer_call_fn_1561616P>?0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_136_layer_call_and_return_conditional_losses_1561627\DE/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_136_layer_call_fn_1561636ODE/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dropout_60_layer_call_and_return_conditional_losses_1561581^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÀ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 ©
G__inference_dropout_60_layer_call_and_return_conditional_losses_1561586^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÀ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 
,__inference_dropout_60_layer_call_fn_1561591Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÀ
p
ª "ÿÿÿÿÿÿÿÿÿÀ
,__inference_dropout_60_layer_call_fn_1561596Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÀ
p 
ª "ÿÿÿÿÿÿÿÿÿÀ¬
I__inference_embedding_60_layer_call_and_return_conditional_losses_1561472_/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd2
 
.__inference_embedding_60_layer_call_fn_1561479R/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿd2Ð
U__inference_global_max_pooling1d_144_layer_call_and_return_conditional_losses_1560800wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¨
:__inference_global_max_pooling1d_144_layer_call_fn_1560806jE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
U__inference_global_max_pooling1d_145_layer_call_and_return_conditional_losses_1560813wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¨
:__inference_global_max_pooling1d_145_layer_call_fn_1560819jE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
U__inference_global_max_pooling1d_146_layer_call_and_return_conditional_losses_1560826wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¨
:__inference_global_max_pooling1d_146_layer_call_fn_1560832jE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
E__inference_model_60_layer_call_and_return_conditional_losses_1561060o$%>?DE9¢6
/¢,
"
input_61ÿÿÿÿÿÿÿÿÿd
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
E__inference_model_60_layer_call_and_return_conditional_losses_1561097o$%>?DE9¢6
/¢,
"
input_61ÿÿÿÿÿÿÿÿÿd
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
E__inference_model_60_layer_call_and_return_conditional_losses_1561339m$%>?DE7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
E__inference_model_60_layer_call_and_return_conditional_losses_1561408m$%>?DE7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_model_60_layer_call_fn_1561162b$%>?DE9¢6
/¢,
"
input_61ÿÿÿÿÿÿÿÿÿd
p

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_model_60_layer_call_fn_1561226b$%>?DE9¢6
/¢,
"
input_61ÿÿÿÿÿÿÿÿÿd
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_model_60_layer_call_fn_1561435`$%>?DE7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_model_60_layer_call_fn_1561462`$%>?DE7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p 

 
ª "ÿÿÿÿÿÿÿÿÿ­
%__inference_signature_wrapper_1561263$%>?DE=¢:
¢ 
3ª0
.
input_61"
input_61ÿÿÿÿÿÿÿÿÿd"5ª2
0
	dense_136# 
	dense_136ÿÿÿÿÿÿÿÿÿ