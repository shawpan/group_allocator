��
�)�)
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
ParseExample

serialized	
names
sparse_keys*Nsparse

dense_keys*Ndense
dense_defaults2Tdense
sparse_indices	*Nsparse
sparse_values2sparse_types
sparse_shapes	*Nsparse
dense_values2Tdense"
Nsparseint("
Ndenseint("%
sparse_types
list(type)(:
2	"
Tdense
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
�
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
z
SparseSegmentMean	
data"T
indices"Tidx
segment_ids
output"T"
Ttype:
2"
Tidxtype0:
2	
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
�
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�
9
VarIsInitializedOp
resource
is_initialized
�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.13.12
b'unknown'8��

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
_class
loc:@global_step*
dtype0	*
_output_shapes
: *
shape: 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
_output_shapes
: *
T0	
o
input_example_tensorPlaceholder*#
_output_shapes
:���������*
shape:���������*
dtype0
U
ParseExample/ConstConst*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_2Const*
dtype0*
_output_shapes
: *
valueB 
W
ParseExample/Const_3Const*
dtype0*
_output_shapes
: *
valueB 
W
ParseExample/Const_4Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_5Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_6Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_7Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_8Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_9Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_10Const*
valueB *
dtype0*
_output_shapes
: 
b
ParseExample/ParseExample/namesConst*
valueB *
dtype0*
_output_shapes
: 
r
'ParseExample/ParseExample/sparse_keys_0Const*
dtype0*
_output_shapes
: *
valueB B
feature_14
r
'ParseExample/ParseExample/sparse_keys_1Const*
valueB B
feature_18*
dtype0*
_output_shapes
: 
q
'ParseExample/ParseExample/sparse_keys_2Const*
valueB B	feature_2*
dtype0*
_output_shapes
: 
q
'ParseExample/ParseExample/sparse_keys_3Const*
_output_shapes
: *
valueB B	feature_7*
dtype0
q
'ParseExample/ParseExample/sparse_keys_4Const*
valueB B	feature_8*
dtype0*
_output_shapes
: 
q
'ParseExample/ParseExample/sparse_keys_5Const*
valueB B	feature_9*
dtype0*
_output_shapes
: 
q
&ParseExample/ParseExample/dense_keys_0Const*
valueB B
feature_10*
dtype0*
_output_shapes
: 
q
&ParseExample/ParseExample/dense_keys_1Const*
_output_shapes
: *
valueB B
feature_11*
dtype0
q
&ParseExample/ParseExample/dense_keys_2Const*
_output_shapes
: *
valueB B
feature_12*
dtype0
q
&ParseExample/ParseExample/dense_keys_3Const*
valueB B
feature_13*
dtype0*
_output_shapes
: 
q
&ParseExample/ParseExample/dense_keys_4Const*
valueB B
feature_15*
dtype0*
_output_shapes
: 
q
&ParseExample/ParseExample/dense_keys_5Const*
valueB B
feature_16*
dtype0*
_output_shapes
: 
q
&ParseExample/ParseExample/dense_keys_6Const*
valueB B
feature_17*
dtype0*
_output_shapes
: 
p
&ParseExample/ParseExample/dense_keys_7Const*
dtype0*
_output_shapes
: *
valueB B	feature_3
p
&ParseExample/ParseExample/dense_keys_8Const*
valueB B	feature_4*
dtype0*
_output_shapes
: 
p
&ParseExample/ParseExample/dense_keys_9Const*
valueB B	feature_5*
dtype0*
_output_shapes
: 
q
'ParseExample/ParseExample/dense_keys_10Const*
valueB B	feature_6*
dtype0*
_output_shapes
: 
�
ParseExample/ParseExampleParseExampleinput_example_tensorParseExample/ParseExample/names'ParseExample/ParseExample/sparse_keys_0'ParseExample/ParseExample/sparse_keys_1'ParseExample/ParseExample/sparse_keys_2'ParseExample/ParseExample/sparse_keys_3'ParseExample/ParseExample/sparse_keys_4'ParseExample/ParseExample/sparse_keys_5&ParseExample/ParseExample/dense_keys_0&ParseExample/ParseExample/dense_keys_1&ParseExample/ParseExample/dense_keys_2&ParseExample/ParseExample/dense_keys_3&ParseExample/ParseExample/dense_keys_4&ParseExample/ParseExample/dense_keys_5&ParseExample/ParseExample/dense_keys_6&ParseExample/ParseExample/dense_keys_7&ParseExample/ParseExample/dense_keys_8&ParseExample/ParseExample/dense_keys_9'ParseExample/ParseExample/dense_keys_10ParseExample/ConstParseExample/Const_1ParseExample/Const_2ParseExample/Const_3ParseExample/Const_4ParseExample/Const_5ParseExample/Const_6ParseExample/Const_7ParseExample/Const_8ParseExample/Const_9ParseExample/Const_10*
Tdense
2*
Ndense*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:::::::���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*
Nsparse*T
dense_shapesD
B:::::::::::*
sparse_types

2
c
ExponentialDecay/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
Z
ExponentialDecay/Cast/xConst*
value
B :�N*
dtype0*
_output_shapes
: 
f
ExponentialDecay/CastCastExponentialDecay/Cast/x*

SrcT0*
_output_shapes
: *

DstT0
^
ExponentialDecay/Cast_1/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
a
ExponentialDecay/Cast_2Castglobal_step/read*

SrcT0	*
_output_shapes
: *

DstT0
t
ExponentialDecay/truedivRealDivExponentialDecay/Cast_2ExponentialDecay/Cast*
T0*
_output_shapes
: 
q
ExponentialDecay/PowPowExponentialDecay/Cast_1/xExponentialDecay/truediv*
_output_shapes
: *
T0
n
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
_output_shapes
: *
T0
�
{dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"~      *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
�
zdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
|dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *��L>*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
seed2.*
dtype0*
_output_shapes

:~*

seed**
T0
�
ydnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMul�dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal|dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
_output_shapes

:~
�
udnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normalAddydnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulzdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
_output_shapes

:~*
T0
�
Xdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0
VariableV2*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:~*
shape
:~
�
_dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/AssignAssignXdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0udnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal*
_output_shapes

:~*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0
�
]dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/readIdentityXdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
_output_shapes

:~
�
{dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"#      *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
�
zdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
dtype0
�
|dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *��j>*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
dtype0*
_output_shapes

:#*

seed**
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
seed27
�
ydnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMul�dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal|dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
_output_shapes

:#
�
udnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normalAddydnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulzdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
_output_shapes

:#
�
Xdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0
VariableV2*
dtype0*
_output_shapes

:#*
shape
:#*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0
�
_dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/AssignAssignXdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0udnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal*
_output_shapes

:#*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0
�
]dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/readIdentityXdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
_output_shapes

:#
�
zdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"�      *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
�
ydnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
{dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *��L>*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalzdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:	�*

seed**
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
seed2@
�
xdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMul�dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	�
�
tdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normalAddxdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulydnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	�
�
Wdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0
VariableV2*
shape:	�*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:	�
�
^dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/AssignAssignWdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0tdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal*
_output_shapes
:	�*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0
�
\dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/readIdentityWdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	�
�
zdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"      *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
�
ydnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
dtype0
�
{dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *.��>*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalzdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*

seed**
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
seed2I*
dtype0*
_output_shapes

:
�
xdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMul�dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes

:*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0
�
tdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normalAddxdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulydnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
_output_shapes

:*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0
�
Wdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0
VariableV2*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:*
shape
:
�
^dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/AssignAssignWdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0tdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
_output_shapes

:
�
\dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/readIdentityWdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
_output_shapes

:*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0
�
zdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0
�
ydnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
{dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *��j>*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalzdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
seed2R*
dtype0*
_output_shapes

:*

seed**
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0
�
xdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMul�dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes

:*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0
�
tdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normalAddxdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulydnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
_output_shapes

:
�
Wdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0
VariableV2*
dtype0*
_output_shapes

:*
shape
:*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0
�
^dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/AssignAssignWdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0tdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
_output_shapes

:
�
\dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/readIdentityWdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
_output_shapes

:*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0
�
zdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"�      *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
�
ydnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
{dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *��L>*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalzdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
seed2[*
dtype0*
_output_shapes
:	�*

seed**
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0
�
xdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMul�dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
_output_shapes
:	�
�
tdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normalAddxdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulydnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
_output_shapes
:	�
�
Wdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0
VariableV2*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:	�*
shape:	�
�
^dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/AssignAssignWdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0tdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal*
_output_shapes
:	�*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0
�
\dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/readIdentityWdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
_output_shapes
:	�*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0
�
;dnn/input_from_feature_columns/input_layer/feature_10/sub/yConst*
valueB
 *�̐N*
dtype0*
_output_shapes
: 
�
9dnn/input_from_feature_columns/input_layer/feature_10/subSubParseExample/ParseExample:18;dnn/input_from_feature_columns/input_layer/feature_10/sub/y*
T0*'
_output_shapes
:���������
�
?dnn/input_from_feature_columns/input_layer/feature_10/truediv/yConst*
_output_shapes
: *
valueB
 *�.�N*
dtype0
�
=dnn/input_from_feature_columns/input_layer/feature_10/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_10/sub?dnn/input_from_feature_columns/input_layer/feature_10/truediv/y*
T0*'
_output_shapes
:���������
�
;dnn/input_from_feature_columns/input_layer/feature_10/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_10/truediv*
T0*
_output_shapes
:
�
Idnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
Kdnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Cdnn/input_from_feature_columns/input_layer/feature_10/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_10/ShapeIdnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
�
Ednn/input_from_feature_columns/input_layer/feature_10/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/feature_10/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_10/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_10/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
=dnn/input_from_feature_columns/input_layer/feature_10/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_10/truedivCdnn/input_from_feature_columns/input_layer/feature_10/Reshape/shape*'
_output_shapes
:���������*
T0
�
;dnn/input_from_feature_columns/input_layer/feature_11/sub/yConst*
valueB
 *��?*
dtype0*
_output_shapes
: 
�
9dnn/input_from_feature_columns/input_layer/feature_11/subSubParseExample/ParseExample:19;dnn/input_from_feature_columns/input_layer/feature_11/sub/y*
T0*'
_output_shapes
:���������
�
?dnn/input_from_feature_columns/input_layer/feature_11/truediv/yConst*
dtype0*
_output_shapes
: *
valueB
 *�?�<
�
=dnn/input_from_feature_columns/input_layer/feature_11/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_11/sub?dnn/input_from_feature_columns/input_layer/feature_11/truediv/y*'
_output_shapes
:���������*
T0
�
;dnn/input_from_feature_columns/input_layer/feature_11/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_11/truediv*
T0*
_output_shapes
:
�
Idnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
�
Kdnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
Cdnn/input_from_feature_columns/input_layer/feature_11/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_11/ShapeIdnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
�
Ednn/input_from_feature_columns/input_layer/feature_11/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/feature_11/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_11/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_11/Reshape/shape/1*
N*
_output_shapes
:*
T0
�
=dnn/input_from_feature_columns/input_layer/feature_11/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_11/truedivCdnn/input_from_feature_columns/input_layer/feature_11/Reshape/shape*'
_output_shapes
:���������*
T0
�
;dnn/input_from_feature_columns/input_layer/feature_12/sub/yConst*
valueB
 *$6�@*
dtype0*
_output_shapes
: 
�
9dnn/input_from_feature_columns/input_layer/feature_12/subSubParseExample/ParseExample:20;dnn/input_from_feature_columns/input_layer/feature_12/sub/y*
T0*'
_output_shapes
:���������
�
?dnn/input_from_feature_columns/input_layer/feature_12/truediv/yConst*
valueB
 *T�A*
dtype0*
_output_shapes
: 
�
=dnn/input_from_feature_columns/input_layer/feature_12/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_12/sub?dnn/input_from_feature_columns/input_layer/feature_12/truediv/y*
T0*'
_output_shapes
:���������
�
;dnn/input_from_feature_columns/input_layer/feature_12/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_12/truediv*
_output_shapes
:*
T0
�
Idnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Cdnn/input_from_feature_columns/input_layer/feature_12/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_12/ShapeIdnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
�
Ednn/input_from_feature_columns/input_layer/feature_12/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/feature_12/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_12/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_12/Reshape/shape/1*
N*
_output_shapes
:*
T0
�
=dnn/input_from_feature_columns/input_layer/feature_12/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_12/truedivCdnn/input_from_feature_columns/input_layer/feature_12/Reshape/shape*'
_output_shapes
:���������*
T0
�
;dnn/input_from_feature_columns/input_layer/feature_13/sub/yConst*
valueB
 *[BD*
dtype0*
_output_shapes
: 
�
9dnn/input_from_feature_columns/input_layer/feature_13/subSubParseExample/ParseExample:21;dnn/input_from_feature_columns/input_layer/feature_13/sub/y*
T0*'
_output_shapes
:���������
�
?dnn/input_from_feature_columns/input_layer/feature_13/truediv/yConst*
valueB
 *DZD*
dtype0*
_output_shapes
: 
�
=dnn/input_from_feature_columns/input_layer/feature_13/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_13/sub?dnn/input_from_feature_columns/input_layer/feature_13/truediv/y*
T0*'
_output_shapes
:���������
�
;dnn/input_from_feature_columns/input_layer/feature_13/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_13/truediv*
T0*
_output_shapes
:
�
Idnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
�
Kdnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
Cdnn/input_from_feature_columns/input_layer/feature_13/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_13/ShapeIdnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
Ednn/input_from_feature_columns/input_layer/feature_13/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/feature_13/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_13/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_13/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
=dnn/input_from_feature_columns/input_layer/feature_13/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_13/truedivCdnn/input_from_feature_columns/input_layer/feature_13/Reshape/shape*
T0*'
_output_shapes
:���������
�
Fdnn/input_from_feature_columns/input_layer/feature_14_embedding/lookupStringToHashBucketFastParseExample/ParseExample:6*#
_output_shapes
:���������*
num_buckets~
�
hdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
gdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SliceSliceParseExample/ParseExample:12hdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice/begingdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
adnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/ProdProdbdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slicebdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Const*
_output_shapes
: *
T0	
�
mdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
�
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
ednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:12mdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2/indicesjdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2/axis*
Tindices0*
Tparams0	*
_output_shapes
: *
Taxis0
�
cdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Cast/xPackadnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Prodednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
�
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExampleParseExample/ParseExample:12cdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Cast/x*-
_output_shapes
:���������:
�
sdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshape/IdentityIdentityFdnn/input_from_feature_columns/input_layer/feature_14_embedding/lookup*
T0	*#
_output_shapes
:���������
�
kdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
idnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GreaterEqualGreaterEqualsdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshape/Identitykdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GreaterEqual/y*#
_output_shapes
:���������*
T0	
�
bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/WhereWhereidnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GreaterEqual*'
_output_shapes
:���������
�
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/ReshapeReshapebdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Wherejdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape/shape*#
_output_shapes
:���������*
T0	
�
ldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
gdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_1GatherV2jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshapeddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_1/axis*'
_output_shapes
:���������*
Taxis0*
Tindices0	*
Tparams0	
�
ldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
gdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_2GatherV2sdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshape/Identityddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_2/axis*#
_output_shapes
:���������*
Taxis0*
Tindices0	*
Tparams0	
�
ednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/IdentityIdentityldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
vdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B	 R *
dtype0	
�
�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsgdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_1gdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_2ednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Identityvdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:���������:���������:���������:���������
�
�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
�
�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
�
�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
end_mask*#
_output_shapes
:���������*
T0	*
Index0*
shrink_axis_mask
�
ydnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/CastCast�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:���������*

DstT0
�
{dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/UniqueUnique�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:���������:���������*
T0	
�
�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*
_output_shapes
: *
value	B : *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0
�
�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2]dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/read{dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/Unique�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0	*
Tparams0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*'
_output_shapes
:���������*
Taxis0
�
�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:���������*
T0
�
tdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparseSparseSegmentMean�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity}dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/Unique:1ydnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:���������
�
ldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
fdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_1Reshape�dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:���������
�
bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/ShapeShapetdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
�
pdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
�
rdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
rdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_sliceStridedSlicebdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Shapepdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stackrdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_1rdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
�
ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
�
bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/stackPackddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/stack/0jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice*
_output_shapes
:*
T0*
N
�
adnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/TileTilefdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_1bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/stack*
T0
*0
_output_shapes
:������������������
�
gdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/zeros_like	ZerosLiketdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
\dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weightsSelectadnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Tilegdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/zeros_liketdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:���������*
T0
�
cdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Cast_1CastParseExample/ParseExample:12*
_output_shapes
:*

DstT0*

SrcT0	
�
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
idnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1Slicecdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Cast_1jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1/beginidnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
�
ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Shape_1Shape\dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights*
_output_shapes
:*
T0
�
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
�
idnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2/sizeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2Sliceddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Shape_1jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2/beginidnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2/size*
_output_shapes
:*
Index0*
T0
�
hdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
cdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/concatConcatV2ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2hdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
�
fdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_2Reshape\dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weightscdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/concat*'
_output_shapes
:���������*
T0
�
Ednn/input_from_feature_columns/input_layer/feature_14_embedding/ShapeShapefdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_2*
T0*
_output_shapes
:
�
Sdnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Udnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
�
Udnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
Mdnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_sliceStridedSliceEdnn/input_from_feature_columns/input_layer/feature_14_embedding/ShapeSdnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stackUdnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stack_1Udnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
�
Odnn/input_from_feature_columns/input_layer/feature_14_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Mdnn/input_from_feature_columns/input_layer/feature_14_embedding/Reshape/shapePackMdnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_sliceOdnn/input_from_feature_columns/input_layer/feature_14_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0
�
Gdnn/input_from_feature_columns/input_layer/feature_14_embedding/ReshapeReshapefdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_2Mdnn/input_from_feature_columns/input_layer/feature_14_embedding/Reshape/shape*
T0*'
_output_shapes
:���������
�
;dnn/input_from_feature_columns/input_layer/feature_15/sub/yConst*
valueB
 *���@*
dtype0*
_output_shapes
: 
�
9dnn/input_from_feature_columns/input_layer/feature_15/subSubParseExample/ParseExample:22;dnn/input_from_feature_columns/input_layer/feature_15/sub/y*
T0*'
_output_shapes
:���������
�
?dnn/input_from_feature_columns/input_layer/feature_15/truediv/yConst*
valueB
 *B&AA*
dtype0*
_output_shapes
: 
�
=dnn/input_from_feature_columns/input_layer/feature_15/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_15/sub?dnn/input_from_feature_columns/input_layer/feature_15/truediv/y*
T0*'
_output_shapes
:���������
�
;dnn/input_from_feature_columns/input_layer/feature_15/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_15/truediv*
_output_shapes
:*
T0
�
Idnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Cdnn/input_from_feature_columns/input_layer/feature_15/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_15/ShapeIdnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
�
Ednn/input_from_feature_columns/input_layer/feature_15/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
�
Cdnn/input_from_feature_columns/input_layer/feature_15/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_15/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_15/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
=dnn/input_from_feature_columns/input_layer/feature_15/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_15/truedivCdnn/input_from_feature_columns/input_layer/feature_15/Reshape/shape*
T0*'
_output_shapes
:���������
�
;dnn/input_from_feature_columns/input_layer/feature_16/sub/yConst*
valueB
 *���A*
dtype0*
_output_shapes
: 
�
9dnn/input_from_feature_columns/input_layer/feature_16/subSubParseExample/ParseExample:23;dnn/input_from_feature_columns/input_layer/feature_16/sub/y*
T0*'
_output_shapes
:���������
�
?dnn/input_from_feature_columns/input_layer/feature_16/truediv/yConst*
dtype0*
_output_shapes
: *
valueB
 *��B
�
=dnn/input_from_feature_columns/input_layer/feature_16/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_16/sub?dnn/input_from_feature_columns/input_layer/feature_16/truediv/y*'
_output_shapes
:���������*
T0
�
;dnn/input_from_feature_columns/input_layer/feature_16/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_16/truediv*
T0*
_output_shapes
:
�
Idnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
�
Kdnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
Cdnn/input_from_feature_columns/input_layer/feature_16/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_16/ShapeIdnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
Ednn/input_from_feature_columns/input_layer/feature_16/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/feature_16/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_16/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_16/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
=dnn/input_from_feature_columns/input_layer/feature_16/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_16/truedivCdnn/input_from_feature_columns/input_layer/feature_16/Reshape/shape*'
_output_shapes
:���������*
T0
�
;dnn/input_from_feature_columns/input_layer/feature_17/sub/yConst*
valueB
 *|��A*
dtype0*
_output_shapes
: 
�
9dnn/input_from_feature_columns/input_layer/feature_17/subSubParseExample/ParseExample:24;dnn/input_from_feature_columns/input_layer/feature_17/sub/y*
T0*'
_output_shapes
:���������
�
?dnn/input_from_feature_columns/input_layer/feature_17/truediv/yConst*
valueB
 *WW�B*
dtype0*
_output_shapes
: 
�
=dnn/input_from_feature_columns/input_layer/feature_17/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_17/sub?dnn/input_from_feature_columns/input_layer/feature_17/truediv/y*'
_output_shapes
:���������*
T0
�
;dnn/input_from_feature_columns/input_layer/feature_17/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_17/truediv*
T0*
_output_shapes
:
�
Idnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Kdnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
Cdnn/input_from_feature_columns/input_layer/feature_17/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_17/ShapeIdnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
�
Ednn/input_from_feature_columns/input_layer/feature_17/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Cdnn/input_from_feature_columns/input_layer/feature_17/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_17/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_17/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
=dnn/input_from_feature_columns/input_layer/feature_17/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_17/truedivCdnn/input_from_feature_columns/input_layer/feature_17/Reshape/shape*
T0*'
_output_shapes
:���������
�
Fdnn/input_from_feature_columns/input_layer/feature_18_embedding/lookupStringToHashBucketFastParseExample/ParseExample:7*
num_buckets#*#
_output_shapes
:���������
�
hdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
�
gdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SliceSliceParseExample/ParseExample:13hdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice/begingdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
�
bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
adnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/ProdProdbdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slicebdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Const*
_output_shapes
: *
T0	
�
mdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
�
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
ednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:13mdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2/indicesjdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
�
cdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Cast/xPackadnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Prodednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
�
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:1ParseExample/ParseExample:13cdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Cast/x*-
_output_shapes
:���������:
�
sdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshape/IdentityIdentityFdnn/input_from_feature_columns/input_layer/feature_18_embedding/lookup*#
_output_shapes
:���������*
T0	
�
kdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
idnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GreaterEqualGreaterEqualsdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshape/Identitykdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:���������
�
bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/WhereWhereidnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GreaterEqual*'
_output_shapes
:���������
�
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/ReshapeReshapebdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Wherejdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape/shape*#
_output_shapes
:���������*
T0	
�
ldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
gdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_1GatherV2jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshapeddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_1/axis*
Tindices0	*
Tparams0	*'
_output_shapes
:���������*
Taxis0
�
ldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
gdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_2GatherV2sdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshape/Identityddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:���������
�
ednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/IdentityIdentityldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
vdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsgdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_1gdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_2ednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Identityvdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:���������:���������:���������:���������*
T0	
�
�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
Index0*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:���������
�
ydnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/CastCast�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:���������*

DstT0
�
{dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/UniqueUnique�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:���������:���������
�
�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2]dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/read{dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/Unique�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*
Tindices0	*
Tparams0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*'
_output_shapes
:���������
�
�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:���������
�
tdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparseSparseSegmentMean�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity}dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/Unique:1ydnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:���������
�
ldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   
�
fdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_1Reshape�dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:���������
�
bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/ShapeShapetdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
�
pdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
rdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
rdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_sliceStridedSlicebdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Shapepdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stackrdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_1rdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
�
ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
�
bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/stackPackddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/stack/0jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
�
adnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/TileTilefdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_1bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/stack*
T0
*0
_output_shapes
:������������������
�
gdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/zeros_like	ZerosLiketdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
\dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weightsSelectadnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Tilegdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/zeros_liketdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:���������*
T0
�
cdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Cast_1CastParseExample/ParseExample:13*
_output_shapes
:*

DstT0*

SrcT0	
�
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
idnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1Slicecdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Cast_1jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1/beginidnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
�
ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Shape_1Shape\dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights*
T0*
_output_shapes
:
�
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
�
idnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
valueB:
���������*
dtype0
�
ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2Sliceddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Shape_1jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2/beginidnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
hdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
cdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/concatConcatV2ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2hdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0
�
fdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_2Reshape\dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weightscdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/concat*
T0*'
_output_shapes
:���������
�
Ednn/input_from_feature_columns/input_layer/feature_18_embedding/ShapeShapefdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_2*
_output_shapes
:*
T0
�
Sdnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Udnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
�
Udnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
Mdnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_sliceStridedSliceEdnn/input_from_feature_columns/input_layer/feature_18_embedding/ShapeSdnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stackUdnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stack_1Udnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
�
Odnn/input_from_feature_columns/input_layer/feature_18_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
�
Mdnn/input_from_feature_columns/input_layer/feature_18_embedding/Reshape/shapePackMdnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_sliceOdnn/input_from_feature_columns/input_layer/feature_18_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
Gdnn/input_from_feature_columns/input_layer/feature_18_embedding/ReshapeReshapefdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_2Mdnn/input_from_feature_columns/input_layer/feature_18_embedding/Reshape/shape*
T0*'
_output_shapes
:���������
�
Ednn/input_from_feature_columns/input_layer/feature_2_embedding/lookupStringToHashBucketFastParseExample/ParseExample:8*#
_output_shapes
:���������*
num_buckets�
�
fdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
ednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SliceSliceParseExample/ParseExample:14fdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice/beginednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
�
`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
_dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/ProdProd`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Const*
T0	*
_output_shapes
: 
�
kdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
cdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:14kdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2/indiceshdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2/axis*
Tindices0*
Tparams0	*
_output_shapes
: *
Taxis0
�
adnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Cast/xPack_dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
�
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:2ParseExample/ParseExample:14adnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Cast/x*-
_output_shapes
:���������:
�
qdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshape/IdentityIdentityEdnn/input_from_feature_columns/input_layer/feature_2_embedding/lookup*#
_output_shapes
:���������*
T0	
�
idnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	
�
gdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GreaterEqualGreaterEqualqdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshape/Identityidnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GreaterEqual/y*#
_output_shapes
:���������*
T0	
�
`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/WhereWheregdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GreaterEqual*'
_output_shapes
:���������
�
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/ReshapeReshape`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Wherehdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape/shape*#
_output_shapes
:���������*
T0	
�
jdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
ednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_1GatherV2hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshapebdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_1/axis*'
_output_shapes
:���������*
Taxis0*
Tindices0	*
Tparams0	
�
jdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
ednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_2GatherV2qdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshape/Identitybdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_2/axis*#
_output_shapes
:���������*
Taxis0*
Tindices0	*
Tparams0	
�
cdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/IdentityIdentityjdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
tdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_1ednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_2cdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Identitytdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:���������:���������:���������:���������
�
�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:���������*
T0	*
Index0
�
wdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/CastCast�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:���������*

DstT0
�
ydnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/UniqueUnique�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:���������:���������
�
�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2\dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/readydnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/Unique�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:���������*
Taxis0*
Tindices0	*
Tparams0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0
�
�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:���������
�
rdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparseSparseSegmentMean�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity{dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/Unique:1wdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:���������*
T0
�
jdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
ddnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_1Reshape�dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2jdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:���������
�
`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/ShapeShaperdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
�
ndnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
pdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
pdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Shapendnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stackpdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_1pdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
�
`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/stackPackbdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/stack/0hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
�
_dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/TileTileddnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_1`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/stack*
T0
*0
_output_shapes
:������������������
�
ednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/zeros_like	ZerosLikerdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:���������*
T0
�
Zdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weightsSelect_dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Tileednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/zeros_likerdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
adnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Cast_1CastParseExample/ParseExample:14*

SrcT0	*
_output_shapes
:*

DstT0
�
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
gdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1Sliceadnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Cast_1hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1/begingdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Shape_1ShapeZdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights*
_output_shapes
:*
T0
�
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
�
gdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2/sizeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2Slicebdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Shape_1hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2/begingdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
fdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
adnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/concatConcatV2bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2fdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/concat/axis*
_output_shapes
:*
T0*
N
�
ddnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_2ReshapeZdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weightsadnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/concat*
T0*'
_output_shapes
:���������
�
Ddnn/input_from_feature_columns/input_layer/feature_2_embedding/ShapeShapeddnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_2*
T0*
_output_shapes
:
�
Rdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Tdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Tdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
Ldnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/feature_2_embedding/ShapeRdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stackTdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
�
Ndnn/input_from_feature_columns/input_layer/feature_2_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ldnn/input_from_feature_columns/input_layer/feature_2_embedding/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_sliceNdnn/input_from_feature_columns/input_layer/feature_2_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
Fdnn/input_from_feature_columns/input_layer/feature_2_embedding/ReshapeReshapeddnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_2Ldnn/input_from_feature_columns/input_layer/feature_2_embedding/Reshape/shape*
T0*'
_output_shapes
:���������

:dnn/input_from_feature_columns/input_layer/feature_3/sub/yConst*
valueB
 *��OD*
dtype0*
_output_shapes
: 
�
8dnn/input_from_feature_columns/input_layer/feature_3/subSubParseExample/ParseExample:25:dnn/input_from_feature_columns/input_layer/feature_3/sub/y*'
_output_shapes
:���������*
T0
�
>dnn/input_from_feature_columns/input_layer/feature_3/truediv/yConst*
valueB
 *{�D*
dtype0*
_output_shapes
: 
�
<dnn/input_from_feature_columns/input_layer/feature_3/truedivRealDiv8dnn/input_from_feature_columns/input_layer/feature_3/sub>dnn/input_from_feature_columns/input_layer/feature_3/truediv/y*'
_output_shapes
:���������*
T0
�
:dnn/input_from_feature_columns/input_layer/feature_3/ShapeShape<dnn/input_from_feature_columns/input_layer/feature_3/truediv*
T0*
_output_shapes
:
�
Hdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Jdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Jdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Bdnn/input_from_feature_columns/input_layer/feature_3/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/feature_3/ShapeHdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stackJdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
�
Ddnn/input_from_feature_columns/input_layer/feature_3/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Bdnn/input_from_feature_columns/input_layer/feature_3/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/feature_3/strided_sliceDdnn/input_from_feature_columns/input_layer/feature_3/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
<dnn/input_from_feature_columns/input_layer/feature_3/ReshapeReshape<dnn/input_from_feature_columns/input_layer/feature_3/truedivBdnn/input_from_feature_columns/input_layer/feature_3/Reshape/shape*
T0*'
_output_shapes
:���������

:dnn/input_from_feature_columns/input_layer/feature_4/sub/yConst*
valueB
 *��EB*
dtype0*
_output_shapes
: 
�
8dnn/input_from_feature_columns/input_layer/feature_4/subSubParseExample/ParseExample:26:dnn/input_from_feature_columns/input_layer/feature_4/sub/y*
T0*'
_output_shapes
:���������
�
>dnn/input_from_feature_columns/input_layer/feature_4/truediv/yConst*
dtype0*
_output_shapes
: *
valueB
 * �B
�
<dnn/input_from_feature_columns/input_layer/feature_4/truedivRealDiv8dnn/input_from_feature_columns/input_layer/feature_4/sub>dnn/input_from_feature_columns/input_layer/feature_4/truediv/y*
T0*'
_output_shapes
:���������
�
:dnn/input_from_feature_columns/input_layer/feature_4/ShapeShape<dnn/input_from_feature_columns/input_layer/feature_4/truediv*
T0*
_output_shapes
:
�
Hdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Jdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
�
Jdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
Bdnn/input_from_feature_columns/input_layer/feature_4/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/feature_4/ShapeHdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stackJdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
Ddnn/input_from_feature_columns/input_layer/feature_4/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Bdnn/input_from_feature_columns/input_layer/feature_4/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/feature_4/strided_sliceDdnn/input_from_feature_columns/input_layer/feature_4/Reshape/shape/1*
_output_shapes
:*
T0*
N
�
<dnn/input_from_feature_columns/input_layer/feature_4/ReshapeReshape<dnn/input_from_feature_columns/input_layer/feature_4/truedivBdnn/input_from_feature_columns/input_layer/feature_4/Reshape/shape*
T0*'
_output_shapes
:���������

:dnn/input_from_feature_columns/input_layer/feature_5/sub/yConst*
valueB
 *�uA*
dtype0*
_output_shapes
: 
�
8dnn/input_from_feature_columns/input_layer/feature_5/subSubParseExample/ParseExample:27:dnn/input_from_feature_columns/input_layer/feature_5/sub/y*'
_output_shapes
:���������*
T0
�
>dnn/input_from_feature_columns/input_layer/feature_5/truediv/yConst*
valueB
 *|?3A*
dtype0*
_output_shapes
: 
�
<dnn/input_from_feature_columns/input_layer/feature_5/truedivRealDiv8dnn/input_from_feature_columns/input_layer/feature_5/sub>dnn/input_from_feature_columns/input_layer/feature_5/truediv/y*'
_output_shapes
:���������*
T0
�
:dnn/input_from_feature_columns/input_layer/feature_5/ShapeShape<dnn/input_from_feature_columns/input_layer/feature_5/truediv*
T0*
_output_shapes
:
�
Hdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Jdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Jdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Bdnn/input_from_feature_columns/input_layer/feature_5/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/feature_5/ShapeHdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stackJdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
�
Ddnn/input_from_feature_columns/input_layer/feature_5/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
�
Bdnn/input_from_feature_columns/input_layer/feature_5/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/feature_5/strided_sliceDdnn/input_from_feature_columns/input_layer/feature_5/Reshape/shape/1*
_output_shapes
:*
T0*
N
�
<dnn/input_from_feature_columns/input_layer/feature_5/ReshapeReshape<dnn/input_from_feature_columns/input_layer/feature_5/truedivBdnn/input_from_feature_columns/input_layer/feature_5/Reshape/shape*
T0*'
_output_shapes
:���������

:dnn/input_from_feature_columns/input_layer/feature_6/sub/yConst*
valueB
 *RG�C*
dtype0*
_output_shapes
: 
�
8dnn/input_from_feature_columns/input_layer/feature_6/subSubParseExample/ParseExample:28:dnn/input_from_feature_columns/input_layer/feature_6/sub/y*
T0*'
_output_shapes
:���������
�
>dnn/input_from_feature_columns/input_layer/feature_6/truediv/yConst*
valueB
 *f �C*
dtype0*
_output_shapes
: 
�
<dnn/input_from_feature_columns/input_layer/feature_6/truedivRealDiv8dnn/input_from_feature_columns/input_layer/feature_6/sub>dnn/input_from_feature_columns/input_layer/feature_6/truediv/y*'
_output_shapes
:���������*
T0
�
:dnn/input_from_feature_columns/input_layer/feature_6/ShapeShape<dnn/input_from_feature_columns/input_layer/feature_6/truediv*
T0*
_output_shapes
:
�
Hdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Jdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
�
Jdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Bdnn/input_from_feature_columns/input_layer/feature_6/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/feature_6/ShapeHdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stackJdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
Ddnn/input_from_feature_columns/input_layer/feature_6/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Bdnn/input_from_feature_columns/input_layer/feature_6/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/feature_6/strided_sliceDdnn/input_from_feature_columns/input_layer/feature_6/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
<dnn/input_from_feature_columns/input_layer/feature_6/ReshapeReshape<dnn/input_from_feature_columns/input_layer/feature_6/truedivBdnn/input_from_feature_columns/input_layer/feature_6/Reshape/shape*
T0*'
_output_shapes
:���������
�
Ednn/input_from_feature_columns/input_layer/feature_7_embedding/lookupStringToHashBucketFastParseExample/ParseExample:9*#
_output_shapes
:���������*
num_buckets
�
fdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
ednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SliceSliceParseExample/ParseExample:15fdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice/beginednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
�
`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
_dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/ProdProd`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Const*
_output_shapes
: *
T0	
�
kdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
cdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:15kdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2/indiceshdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2/axis*
_output_shapes
: *
Taxis0*
Tindices0*
Tparams0	
�
adnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Cast/xPack_dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
�
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:3ParseExample/ParseExample:15adnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Cast/x*-
_output_shapes
:���������:
�
qdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshape/IdentityIdentityEdnn/input_from_feature_columns/input_layer/feature_7_embedding/lookup*#
_output_shapes
:���������*
T0	
�
idnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
gdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GreaterEqualGreaterEqualqdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshape/Identityidnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GreaterEqual/y*#
_output_shapes
:���������*
T0	
�
`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/WhereWheregdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GreaterEqual*'
_output_shapes
:���������
�
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/ReshapeReshape`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Wherehdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape/shape*#
_output_shapes
:���������*
T0	
�
jdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
ednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_1GatherV2hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshapebdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������
�
jdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
ednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_2GatherV2qdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshape/Identitybdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:���������*
Taxis0
�
cdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/IdentityIdentityjdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
tdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_1ednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_2cdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Identitytdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:���������:���������:���������:���������
�
�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:���������*
T0	*
Index0
�
wdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/CastCast�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:���������*

DstT0
�
ydnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/UniqueUnique�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:���������:���������
�
�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2\dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/readydnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/Unique�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0	*
Tparams0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*'
_output_shapes
:���������*
Taxis0
�
�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:���������
�
rdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparseSparseSegmentMean�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity{dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/Unique:1wdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:���������
�
jdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
valueB"����   *
dtype0
�
ddnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_1Reshape�dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2jdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:���������
�
`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/ShapeShaperdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
�
ndnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
pdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
pdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Shapendnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stackpdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_1pdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
�
`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/stackPackbdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/stack/0hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice*
_output_shapes
:*
T0*
N
�
_dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/TileTileddnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_1`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/stack*
T0
*0
_output_shapes
:������������������
�
ednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/zeros_like	ZerosLikerdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
Zdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weightsSelect_dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Tileednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/zeros_likerdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:���������*
T0
�
adnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Cast_1CastParseExample/ParseExample:15*

SrcT0	*
_output_shapes
:*

DstT0
�
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
gdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1Sliceadnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Cast_1hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1/begingdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Shape_1ShapeZdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights*
T0*
_output_shapes
:
�
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
�
gdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2/sizeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2Slicebdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Shape_1hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2/begingdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
fdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
adnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/concatConcatV2bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2fdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
�
ddnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_2ReshapeZdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weightsadnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/concat*
T0*'
_output_shapes
:���������
�
Ddnn/input_from_feature_columns/input_layer/feature_7_embedding/ShapeShapeddnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_2*
_output_shapes
:*
T0
�
Rdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
�
Tdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Tdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ldnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/feature_7_embedding/ShapeRdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stackTdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
�
Ndnn/input_from_feature_columns/input_layer/feature_7_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ldnn/input_from_feature_columns/input_layer/feature_7_embedding/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_sliceNdnn/input_from_feature_columns/input_layer/feature_7_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
Fdnn/input_from_feature_columns/input_layer/feature_7_embedding/ReshapeReshapeddnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_2Ldnn/input_from_feature_columns/input_layer/feature_7_embedding/Reshape/shape*
T0*'
_output_shapes
:���������
�
Ednn/input_from_feature_columns/input_layer/feature_8_embedding/lookupStringToHashBucketFastParseExample/ParseExample:10*
num_buckets*#
_output_shapes
:���������
�
fdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
ednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SliceSliceParseExample/ParseExample:16fdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice/beginednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
�
`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
_dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/ProdProd`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Const*
_output_shapes
: *
T0	
�
kdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
cdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:16kdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2/indiceshdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
�
adnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Cast/xPack_dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N
�
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:4ParseExample/ParseExample:16adnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Cast/x*-
_output_shapes
:���������:
�
qdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshape/IdentityIdentityEdnn/input_from_feature_columns/input_layer/feature_8_embedding/lookup*
T0	*#
_output_shapes
:���������
�
idnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
gdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GreaterEqualGreaterEqualqdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshape/Identityidnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GreaterEqual/y*#
_output_shapes
:���������*
T0	
�
`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/WhereWheregdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GreaterEqual*'
_output_shapes
:���������
�
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/ReshapeReshape`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Wherehdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:���������
�
jdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
ednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_1GatherV2hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshapebdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_1/axis*'
_output_shapes
:���������*
Taxis0*
Tindices0	*
Tparams0	
�
jdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
ednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_2GatherV2qdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshape/Identitybdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:���������*
Taxis0
�
cdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/IdentityIdentityjdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
�
tdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B	 R *
dtype0	
�
�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_1ednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_2cdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Identitytdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:���������:���������:���������:���������*
T0	
�
�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
�
�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
Index0*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:���������
�
wdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/CastCast�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:���������*

DstT0
�
ydnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/UniqueUnique�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:���������:���������
�
�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
�
�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2\dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/readydnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/Unique�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*'
_output_shapes
:���������*
Taxis0*
Tindices0	*
Tparams0
�
�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:���������*
T0
�
rdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparseSparseSegmentMean�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity{dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/Unique:1wdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:���������*
T0
�
jdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
ddnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_1Reshape�dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2jdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:���������
�
`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/ShapeShaperdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
�
ndnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
pdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
�
pdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Shapendnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stackpdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_1pdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
�
bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
�
`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/stackPackbdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/stack/0hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice*
N*
_output_shapes
:*
T0
�
_dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/TileTileddnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_1`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/stack*
T0
*0
_output_shapes
:������������������
�
ednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/zeros_like	ZerosLikerdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
Zdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weightsSelect_dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Tileednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/zeros_likerdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:���������*
T0
�
adnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Cast_1CastParseExample/ParseExample:16*

SrcT0	*
_output_shapes
:*

DstT0
�
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
gdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1Sliceadnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Cast_1hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1/begingdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Shape_1ShapeZdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights*
_output_shapes
:*
T0
�
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
�
gdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2Slicebdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Shape_1hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2/begingdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
fdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
adnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/concatConcatV2bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2fdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
�
ddnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_2ReshapeZdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weightsadnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/concat*
T0*'
_output_shapes
:���������
�
Ddnn/input_from_feature_columns/input_layer/feature_8_embedding/ShapeShapeddnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_2*
T0*
_output_shapes
:
�
Rdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Tdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Tdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ldnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/feature_8_embedding/ShapeRdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stackTdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
�
Ndnn/input_from_feature_columns/input_layer/feature_8_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
�
Ldnn/input_from_feature_columns/input_layer/feature_8_embedding/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_sliceNdnn/input_from_feature_columns/input_layer/feature_8_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
Fdnn/input_from_feature_columns/input_layer/feature_8_embedding/ReshapeReshapeddnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_2Ldnn/input_from_feature_columns/input_layer/feature_8_embedding/Reshape/shape*'
_output_shapes
:���������*
T0
�
Ednn/input_from_feature_columns/input_layer/feature_9_embedding/lookupStringToHashBucketFastParseExample/ParseExample:11*#
_output_shapes
:���������*
num_buckets�
�
fdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
ednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
�
`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SliceSliceParseExample/ParseExample:17fdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice/beginednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
�
`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
_dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/ProdProd`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Const*
_output_shapes
: *
T0	
�
kdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
cdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:17kdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2/indiceshdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
�
adnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Cast/xPack_dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
�
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:5ParseExample/ParseExample:17adnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Cast/x*-
_output_shapes
:���������:
�
qdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshape/IdentityIdentityEdnn/input_from_feature_columns/input_layer/feature_9_embedding/lookup*#
_output_shapes
:���������*
T0	
�
idnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
gdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GreaterEqualGreaterEqualqdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshape/Identityidnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:���������
�
`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/WhereWheregdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GreaterEqual*'
_output_shapes
:���������
�
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/ReshapeReshape`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Wherehdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape/shape*#
_output_shapes
:���������*
T0	
�
jdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
ednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_1GatherV2hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshapebdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:���������
�
jdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
ednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_2GatherV2qdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshape/Identitybdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_2/axis*#
_output_shapes
:���������*
Taxis0*
Tindices0	*
Tparams0	
�
cdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/IdentityIdentityjdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
�
tdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B	 R *
dtype0	
�
�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_1ednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_2cdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Identitytdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:���������:���������:���������:���������
�
�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
�
�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
�
�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
�
�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:���������*
T0	*
Index0
�
wdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/CastCast�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:���������*

DstT0*

SrcT0	
�
ydnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/UniqueUnique�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:���������:���������
�
�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
value	B : *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
dtype0
�
�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2\dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/readydnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/Unique�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:���������*
Taxis0*
Tindices0	*
Tparams0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0
�
�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:���������
�
rdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparseSparseSegmentMean�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity{dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/Unique:1wdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:���������*
T0
�
jdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
valueB"����   *
dtype0
�
ddnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_1Reshape�dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2jdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_1/shape*'
_output_shapes
:���������*
T0

�
`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/ShapeShaperdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
�
ndnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
�
pdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
pdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Shapendnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stackpdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_1pdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
�
bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
�
`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/stackPackbdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/stack/0hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
�
_dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/TileTileddnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_1`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/stack*
T0
*0
_output_shapes
:������������������
�
ednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/zeros_like	ZerosLikerdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
Zdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weightsSelect_dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Tileednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/zeros_likerdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:���������
�
adnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Cast_1CastParseExample/ParseExample:17*
_output_shapes
:*

DstT0*

SrcT0	
�
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
gdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1Sliceadnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Cast_1hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1/begingdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
�
bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Shape_1ShapeZdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights*
T0*
_output_shapes
:
�
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
�
gdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2/sizeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2Slicebdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Shape_1hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2/begingdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
�
fdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
adnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/concatConcatV2bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2fdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/concat/axis*
_output_shapes
:*
T0*
N
�
ddnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_2ReshapeZdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weightsadnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/concat*'
_output_shapes
:���������*
T0
�
Ddnn/input_from_feature_columns/input_layer/feature_9_embedding/ShapeShapeddnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_2*
T0*
_output_shapes
:
�
Rdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Tdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Tdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ldnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/feature_9_embedding/ShapeRdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stackTdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
�
Ndnn/input_from_feature_columns/input_layer/feature_9_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ldnn/input_from_feature_columns/input_layer/feature_9_embedding/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_sliceNdnn/input_from_feature_columns/input_layer/feature_9_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
Fdnn/input_from_feature_columns/input_layer/feature_9_embedding/ReshapeReshapeddnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_2Ldnn/input_from_feature_columns/input_layer/feature_9_embedding/Reshape/shape*
T0*'
_output_shapes
:���������
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
�

1dnn/input_from_feature_columns/input_layer/concatConcatV2=dnn/input_from_feature_columns/input_layer/feature_10/Reshape=dnn/input_from_feature_columns/input_layer/feature_11/Reshape=dnn/input_from_feature_columns/input_layer/feature_12/Reshape=dnn/input_from_feature_columns/input_layer/feature_13/ReshapeGdnn/input_from_feature_columns/input_layer/feature_14_embedding/Reshape=dnn/input_from_feature_columns/input_layer/feature_15/Reshape=dnn/input_from_feature_columns/input_layer/feature_16/Reshape=dnn/input_from_feature_columns/input_layer/feature_17/ReshapeGdnn/input_from_feature_columns/input_layer/feature_18_embedding/ReshapeFdnn/input_from_feature_columns/input_layer/feature_2_embedding/Reshape<dnn/input_from_feature_columns/input_layer/feature_3/Reshape<dnn/input_from_feature_columns/input_layer/feature_4/Reshape<dnn/input_from_feature_columns/input_layer/feature_5/Reshape<dnn/input_from_feature_columns/input_layer/feature_6/ReshapeFdnn/input_from_feature_columns/input_layer/feature_7_embedding/ReshapeFdnn/input_from_feature_columns/input_layer/feature_8_embedding/ReshapeFdnn/input_from_feature_columns/input_layer/feature_9_embedding/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
T0*
N*(
_output_shapes
:����������
�
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"�      *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *MP�*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *MP>*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
�
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
seed2�*
dtype0*
_output_shapes
:	�*

seed*
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
�
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	�
�
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	�*
T0
�
dnn/hiddenlayer_0/kernel/part_0VarHandleOp*
_output_shapes
: *
shape:	�*0
shared_name!dnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0
�
@dnn/hiddenlayer_0/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
�
&dnn/hiddenlayer_0/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
�
3dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:	�
�
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
�
dnn/hiddenlayer_0/bias/part_0VarHandleOp*.
shared_namednn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: *
shape:
�
>dnn/hiddenlayer_0/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
: 
�
$dnn/hiddenlayer_0/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0
�
1dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
:*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0
�
'dnn/hiddenlayer_0/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:	�
w
dnn/hiddenlayer_0/kernelIdentity'dnn/hiddenlayer_0/kernel/ReadVariableOp*
_output_shapes
:	�*
T0
�
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*'
_output_shapes
:���������*
T0

%dnn/hiddenlayer_0/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
n
dnn/hiddenlayer_0/biasIdentity%dnn/hiddenlayer_0/bias/ReadVariableOp*
T0*
_output_shapes
:
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*'
_output_shapes
:���������
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*'
_output_shapes
:���������*
T0
�
;dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Initializer/onesConst*
dtype0*
_output_shapes
:*
valueB*  �?*=
_class3
1/loc:@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0
�
*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0VarHandleOp*=
_class3
1/loc:@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
dtype0*
_output_shapes
: *
shape:*;
shared_name,*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0
�
Kdnn/hiddenlayer_0/batchnorm_0/gamma/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
_output_shapes
: 
�
1dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/AssignAssignVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0;dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Initializer/ones*=
_class3
1/loc:@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
dtype0
�
>dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Read/ReadVariableOpReadVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*=
_class3
1/loc:@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
dtype0*
_output_shapes
:
�
;dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
dtype0*
_output_shapes
:
�
)dnn/hiddenlayer_0/batchnorm_0/beta/part_0VarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
dtype0*
_output_shapes
: *
shape:*:
shared_name+)dnn/hiddenlayer_0/batchnorm_0/beta/part_0
�
Jdnn/hiddenlayer_0/batchnorm_0/beta/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
_output_shapes
: 
�
0dnn/hiddenlayer_0/batchnorm_0/beta/part_0/AssignAssignVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0;dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Initializer/zeros*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
dtype0
�
=dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
dtype0*
_output_shapes
:
�
;dnn/hiddenlayer_0/batchnorm_0/moving_mean/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/moving_mean*
dtype0*
_output_shapes
:
�
)dnn/hiddenlayer_0/batchnorm_0/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape:*:
shared_name+)dnn/hiddenlayer_0/batchnorm_0/moving_mean*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/moving_mean
�
Jdnn/hiddenlayer_0/batchnorm_0/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean*
_output_shapes
: 
�
0dnn/hiddenlayer_0/batchnorm_0/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean;dnn/hiddenlayer_0/batchnorm_0/moving_mean/Initializer/zeros*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/moving_mean*
dtype0
�
=dnn/hiddenlayer_0/batchnorm_0/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/moving_mean*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_0/batchnorm_0/moving_variance/Initializer/onesConst*
valueB*  �?*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/moving_variance*
dtype0*
_output_shapes
:
�
-dnn/hiddenlayer_0/batchnorm_0/moving_varianceVarHandleOp*
shape:*>
shared_name/-dnn/hiddenlayer_0/batchnorm_0/moving_variance*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/moving_variance*
dtype0*
_output_shapes
: 
�
Ndnn/hiddenlayer_0/batchnorm_0/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance*
_output_shapes
: 
�
4dnn/hiddenlayer_0/batchnorm_0/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance>dnn/hiddenlayer_0/batchnorm_0/moving_variance/Initializer/ones*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/moving_variance*
dtype0
�
Adnn/hiddenlayer_0/batchnorm_0/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/moving_variance*
dtype0*
_output_shapes
:
�
1dnn/hiddenlayer_0/batchnorm_0/beta/ReadVariableOpReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
_output_shapes
:*
dtype0
�
"dnn/hiddenlayer_0/batchnorm_0/betaIdentity1dnn/hiddenlayer_0/batchnorm_0/beta/ReadVariableOp*
_output_shapes
:*
T0
�
6dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance*
dtype0*
_output_shapes
:
r
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
+dnn/hiddenlayer_0/batchnorm_0/batchnorm/addAdd6dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOp-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add/y*
T0*
_output_shapes
:
�
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_0/batchnorm_0/batchnorm/add*
T0*
_output_shapes
:
�
2dnn/hiddenlayer_0/batchnorm_0/gamma/ReadVariableOpReadVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
dtype0*
_output_shapes
:
�
#dnn/hiddenlayer_0/batchnorm_0/gammaIdentity2dnn/hiddenlayer_0/batchnorm_0/gamma/ReadVariableOp*
_output_shapes
:*
T0
�
+dnn/hiddenlayer_0/batchnorm_0/batchnorm/mulMul-dnn/hiddenlayer_0/batchnorm_0/batchnorm/Rsqrt#dnn/hiddenlayer_0/batchnorm_0/gamma*
T0*
_output_shapes
:
�
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_1Muldnn/hiddenlayer_0/Relu+dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul*'
_output_shapes
:���������*
T0
�
8dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean*
dtype0*
_output_shapes
:
�
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_2Mul8dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul*
_output_shapes
:*
T0
�
+dnn/hiddenlayer_0/batchnorm_0/batchnorm/subSub"dnn/hiddenlayer_0/batchnorm_0/beta-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_2*
T0*
_output_shapes
:
�
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1Add-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_1+dnn/hiddenlayer_0/batchnorm_0/batchnorm/sub*'
_output_shapes
:���������*
T0
~
dnn/zero_fraction/SizeSize-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1*
_output_shapes
: *
T0*
out_type0	
c
dnn/zero_fraction/LessEqual/yConst*
valueB	 R����*
dtype0	*
_output_shapes
: 
�
dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
�
dnn/zero_fraction/cond/SwitchSwitchdnn/zero_fraction/LessEqualdnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
m
dnn/zero_fraction/cond/switch_tIdentitydnn/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
k
dnn/zero_fraction/cond/switch_fIdentitydnn/zero_fraction/cond/Switch*
_output_shapes
: *
T0

h
dnn/zero_fraction/cond/pred_idIdentitydnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
�
*dnn/zero_fraction/cond/count_nonzero/zerosConst ^dnn/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
-dnn/zero_fraction/cond/count_nonzero/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1*dnn/zero_fraction/cond/count_nonzero/zeros*
T0*'
_output_shapes
:���������
�
4dnn/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitch-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1dnn/zero_fraction/cond/pred_id*
T0*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1*:
_output_shapes(
&:���������:���������
�
)dnn/zero_fraction/cond/count_nonzero/CastCast-dnn/zero_fraction/cond/count_nonzero/NotEqual*'
_output_shapes
:���������*

DstT0*

SrcT0

�
*dnn/zero_fraction/cond/count_nonzero/ConstConst ^dnn/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
�
2dnn/zero_fraction/cond/count_nonzero/nonzero_countSum)dnn/zero_fraction/cond/count_nonzero/Cast*dnn/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
�
dnn/zero_fraction/cond/CastCast2dnn/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
�
,dnn/zero_fraction/cond/count_nonzero_1/zerosConst ^dnn/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/dnn/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch,dnn/zero_fraction/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:���������
�
6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1dnn/zero_fraction/cond/pred_id*
T0*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1*:
_output_shapes(
&:���������:���������
�
+dnn/zero_fraction/cond/count_nonzero_1/CastCast/dnn/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:���������*

DstT0	
�
,dnn/zero_fraction/cond/count_nonzero_1/ConstConst ^dnn/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
�
4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countSum+dnn/zero_fraction/cond/count_nonzero_1/Cast,dnn/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
�
dnn/zero_fraction/cond/MergeMerge4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countdnn/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
�
(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
�
)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
{
+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
�
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
�
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
T0*
_output_shapes
: 
�
$dnn/dnn/hiddenlayer_0/activation/tagConst*
dtype0*
_output_shapes
: *1
value(B& B dnn/dnn/hiddenlayer_0/activation
�
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tag-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1*
_output_shapes
: 
�
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *׳ݾ*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *׳�>*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
�
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
seed2�*
dtype0*
_output_shapes

:*

seed**
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
�
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes

:*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
�
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:
�
dnn/hiddenlayer_1/kernel/part_0VarHandleOp*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: *
shape
:*0
shared_name!dnn/hiddenlayer_1/kernel/part_0
�
@dnn/hiddenlayer_1/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
�
&dnn/hiddenlayer_1/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0
�
3dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:
�
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
�
dnn/hiddenlayer_1/bias/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape:*.
shared_namednn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
�
>dnn/hiddenlayer_1/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
�
$dnn/hiddenlayer_1/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0
�
1dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
�
'dnn/hiddenlayer_1/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:
v
dnn/hiddenlayer_1/kernelIdentity'dnn/hiddenlayer_1/kernel/ReadVariableOp*
_output_shapes

:*
T0
�
dnn/hiddenlayer_1/MatMulMatMul-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1dnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:���������

%dnn/hiddenlayer_1/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
n
dnn/hiddenlayer_1/biasIdentity%dnn/hiddenlayer_1/bias/ReadVariableOp*
T0*
_output_shapes
:
�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*'
_output_shapes
:���������
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:���������
�
;dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Initializer/onesConst*
valueB*  �?*=
_class3
1/loc:@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
dtype0*
_output_shapes
:
�
*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0VarHandleOp*
shape:*;
shared_name,*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*=
_class3
1/loc:@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
dtype0*
_output_shapes
: 
�
Kdnn/hiddenlayer_1/batchnorm_1/gamma/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
_output_shapes
: 
�
1dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/AssignAssignVariableOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0;dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Initializer/ones*=
_class3
1/loc:@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
dtype0
�
>dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Read/ReadVariableOpReadVariableOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*=
_class3
1/loc:@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
dtype0*
_output_shapes
:
�
;dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0*
_output_shapes
:
�
)dnn/hiddenlayer_1/batchnorm_1/beta/part_0VarHandleOp*
shape:*:
shared_name+)dnn/hiddenlayer_1/batchnorm_1/beta/part_0*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0*
_output_shapes
: 
�
Jdnn/hiddenlayer_1/batchnorm_1/beta/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
_output_shapes
: 
�
0dnn/hiddenlayer_1/batchnorm_1/beta/part_0/AssignAssignVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0;dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Initializer/zeros*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0
�
=dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0*
_output_shapes
:
�
;dnn/hiddenlayer_1/batchnorm_1/moving_mean/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/moving_mean*
dtype0*
_output_shapes
:
�
)dnn/hiddenlayer_1/batchnorm_1/moving_meanVarHandleOp*
_output_shapes
: *
shape:*:
shared_name+)dnn/hiddenlayer_1/batchnorm_1/moving_mean*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/moving_mean*
dtype0
�
Jdnn/hiddenlayer_1/batchnorm_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean*
_output_shapes
: 
�
0dnn/hiddenlayer_1/batchnorm_1/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean;dnn/hiddenlayer_1/batchnorm_1/moving_mean/Initializer/zeros*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/moving_mean*
dtype0
�
=dnn/hiddenlayer_1/batchnorm_1/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/moving_mean*
dtype0*
_output_shapes
:
�
>dnn/hiddenlayer_1/batchnorm_1/moving_variance/Initializer/onesConst*
_output_shapes
:*
valueB*  �?*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/moving_variance*
dtype0
�
-dnn/hiddenlayer_1/batchnorm_1/moving_varianceVarHandleOp*
shape:*>
shared_name/-dnn/hiddenlayer_1/batchnorm_1/moving_variance*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/moving_variance*
dtype0*
_output_shapes
: 
�
Ndnn/hiddenlayer_1/batchnorm_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance*
_output_shapes
: 
�
4dnn/hiddenlayer_1/batchnorm_1/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance>dnn/hiddenlayer_1/batchnorm_1/moving_variance/Initializer/ones*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/moving_variance*
dtype0
�
Adnn/hiddenlayer_1/batchnorm_1/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/moving_variance*
dtype0*
_output_shapes
:
�
1dnn/hiddenlayer_1/batchnorm_1/beta/ReadVariableOpReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0*
_output_shapes
:
�
"dnn/hiddenlayer_1/batchnorm_1/betaIdentity1dnn/hiddenlayer_1/batchnorm_1/beta/ReadVariableOp*
T0*
_output_shapes
:
�
6dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance*
dtype0*
_output_shapes
:
r
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
+dnn/hiddenlayer_1/batchnorm_1/batchnorm/addAdd6dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOp-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add/y*
_output_shapes
:*
T0
�
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_1/batchnorm_1/batchnorm/add*
_output_shapes
:*
T0
�
2dnn/hiddenlayer_1/batchnorm_1/gamma/ReadVariableOpReadVariableOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
dtype0*
_output_shapes
:
�
#dnn/hiddenlayer_1/batchnorm_1/gammaIdentity2dnn/hiddenlayer_1/batchnorm_1/gamma/ReadVariableOp*
_output_shapes
:*
T0
�
+dnn/hiddenlayer_1/batchnorm_1/batchnorm/mulMul-dnn/hiddenlayer_1/batchnorm_1/batchnorm/Rsqrt#dnn/hiddenlayer_1/batchnorm_1/gamma*
T0*
_output_shapes
:
�
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_1Muldnn/hiddenlayer_1/Relu+dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul*'
_output_shapes
:���������*
T0
�
8dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean*
dtype0*
_output_shapes
:
�
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_2Mul8dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul*
_output_shapes
:*
T0
�
+dnn/hiddenlayer_1/batchnorm_1/batchnorm/subSub"dnn/hiddenlayer_1/batchnorm_1/beta-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_2*
T0*
_output_shapes
:
�
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1Add-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_1+dnn/hiddenlayer_1/batchnorm_1/batchnorm/sub*'
_output_shapes
:���������*
T0
�
dnn/zero_fraction_1/SizeSize-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1*
_output_shapes
: *
T0*
out_type0	
e
dnn/zero_fraction_1/LessEqual/yConst*
_output_shapes
: *
valueB	 R����*
dtype0	
�
dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 
�
dnn/zero_fraction_1/cond/SwitchSwitchdnn/zero_fraction_1/LessEqualdnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_1/cond/switch_tIdentity!dnn/zero_fraction_1/cond/Switch:1*
_output_shapes
: *
T0

o
!dnn/zero_fraction_1/cond/switch_fIdentitydnn/zero_fraction_1/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_1/cond/pred_idIdentitydnn/zero_fraction_1/LessEqual*
_output_shapes
: *
T0

�
,dnn/zero_fraction_1/cond/count_nonzero/zerosConst"^dnn/zero_fraction_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/dnn/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_1/cond/count_nonzero/zeros*
T0*'
_output_shapes
:���������
�
6dnn/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitch-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1 dnn/zero_fraction_1/cond/pred_id*
T0*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1*:
_output_shapes(
&:���������:���������
�
+dnn/zero_fraction_1/cond/count_nonzero/CastCast/dnn/zero_fraction_1/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:���������*

DstT0
�
,dnn/zero_fraction_1/cond/count_nonzero/ConstConst"^dnn/zero_fraction_1/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
�
4dnn/zero_fraction_1/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_1/cond/count_nonzero/Cast,dnn/zero_fraction_1/cond/count_nonzero/Const*
T0*
_output_shapes
: 
�
dnn/zero_fraction_1/cond/CastCast4dnn/zero_fraction_1/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
�
.dnn/zero_fraction_1/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_1/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_1/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:���������
�
8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitch-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1 dnn/zero_fraction_1/cond/pred_id*:
_output_shapes(
&:���������:���������*
T0*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1
�
-dnn/zero_fraction_1/cond/count_nonzero_1/CastCast1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:���������*

DstT0	
�
.dnn/zero_fraction_1/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_1/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"       
�
6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_1/cond/count_nonzero_1/Cast.dnn/zero_fraction_1/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
�
dnn/zero_fraction_1/cond/MergeMerge6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_1/cond/Cast*
T0	*
N*
_output_shapes
: : 
�
*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Sizednn/zero_fraction_1/cond/Merge*
T0	*
_output_shapes
: 
�
+dnn/zero_fraction_1/counts_to_fraction/CastCast*dnn/zero_fraction_1/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0

-dnn/zero_fraction_1/counts_to_fraction/Cast_1Castdnn/zero_fraction_1/Size*
_output_shapes
: *

DstT0*

SrcT0	
�
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 
�
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
_output_shapes
: *
T0
�
$dnn/dnn/hiddenlayer_1/activation/tagConst*
dtype0*
_output_shapes
: *1
value(B& B dnn/dnn/hiddenlayer_1/activation
�
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tag-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1*
_output_shapes
: 
�
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *0�*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *0?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
�
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2�*
dtype0*
_output_shapes

:*

seed*
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: *
T0
�
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
�
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
�
dnn/logits/kernel/part_0VarHandleOp*
shape
:*)
shared_namednn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
�
9dnn/logits/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel/part_0*
_output_shapes
: 
�
dnn/logits/kernel/part_0/AssignAssignVariableOpdnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
�
,dnn/logits/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:
�
(dnn/logits/bias/part_0/Initializer/zerosConst*
valueB*    *)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
�
dnn/logits/bias/part_0VarHandleOp*'
shared_namednn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: *
shape:
}
7dnn/logits/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias/part_0*
_output_shapes
: 
�
dnn/logits/bias/part_0/AssignAssignVariableOpdnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*)
_class
loc:@dnn/logits/bias/part_0*
dtype0
�
*dnn/logits/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
y
 dnn/logits/kernel/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:
h
dnn/logits/kernelIdentity dnn/logits/kernel/ReadVariableOp*
T0*
_output_shapes

:
�
dnn/logits/MatMulMatMul-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1dnn/logits/kernel*'
_output_shapes
:���������*
T0
q
dnn/logits/bias/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
`
dnn/logits/biasIdentitydnn/logits/bias/ReadVariableOp*
_output_shapes
:*
T0
s
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*'
_output_shapes
:���������
e
dnn/zero_fraction_2/SizeSizednn/logits/BiasAdd*
T0*
out_type0	*
_output_shapes
: 
e
dnn/zero_fraction_2/LessEqual/yConst*
valueB	 R����*
dtype0	*
_output_shapes
: 
�
dnn/zero_fraction_2/LessEqual	LessEqualdnn/zero_fraction_2/Sizednn/zero_fraction_2/LessEqual/y*
T0	*
_output_shapes
: 
�
dnn/zero_fraction_2/cond/SwitchSwitchdnn/zero_fraction_2/LessEqualdnn/zero_fraction_2/LessEqual*
_output_shapes
: : *
T0

q
!dnn/zero_fraction_2/cond/switch_tIdentity!dnn/zero_fraction_2/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_2/cond/switch_fIdentitydnn/zero_fraction_2/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_2/cond/pred_idIdentitydnn/zero_fraction_2/LessEqual*
T0
*
_output_shapes
: 
�
,dnn/zero_fraction_2/cond/count_nonzero/zerosConst"^dnn/zero_fraction_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/dnn/zero_fraction_2/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_2/cond/count_nonzero/zeros*'
_output_shapes
:���������*
T0
�
6dnn/zero_fraction_2/cond/count_nonzero/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_2/cond/pred_id*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:���������:���������*
T0
�
+dnn/zero_fraction_2/cond/count_nonzero/CastCast/dnn/zero_fraction_2/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:���������*

DstT0
�
,dnn/zero_fraction_2/cond/count_nonzero/ConstConst"^dnn/zero_fraction_2/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
�
4dnn/zero_fraction_2/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_2/cond/count_nonzero/Cast,dnn/zero_fraction_2/cond/count_nonzero/Const*
_output_shapes
: *
T0
�
dnn/zero_fraction_2/cond/CastCast4dnn/zero_fraction_2/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
�
.dnn/zero_fraction_2/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_2/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_2/cond/count_nonzero_1/zeros*'
_output_shapes
:���������*
T0
�
8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_2/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:���������:���������
�
-dnn/zero_fraction_2/cond/count_nonzero_1/CastCast1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual*'
_output_shapes
:���������*

DstT0	*

SrcT0

�
.dnn/zero_fraction_2/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_2/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
�
6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_2/cond/count_nonzero_1/Cast.dnn/zero_fraction_2/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
�
dnn/zero_fraction_2/cond/MergeMerge6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_2/cond/Cast*
T0	*
N*
_output_shapes
: : 
�
*dnn/zero_fraction_2/counts_to_fraction/subSubdnn/zero_fraction_2/Sizednn/zero_fraction_2/cond/Merge*
_output_shapes
: *
T0	
�
+dnn/zero_fraction_2/counts_to_fraction/CastCast*dnn/zero_fraction_2/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0

-dnn/zero_fraction_2/counts_to_fraction/Cast_1Castdnn/zero_fraction_2/Size*
_output_shapes
: *

DstT0*

SrcT0	
�
.dnn/zero_fraction_2/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_2/counts_to_fraction/Cast-dnn/zero_fraction_2/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_2/fractionIdentity.dnn/zero_fraction_2/counts_to_fraction/truediv*
_output_shapes
: *
T0
�
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
�
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/fraction*
_output_shapes
: *
T0
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
x
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
W
dnn/head/logits/ShapeShapednn/logits/BiasAdd*
T0*
_output_shapes
:
k
)dnn/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
[
Sdnn/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
shape: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
~
save/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
_output_shapes
:*
dtype0
X
save/IdentityIdentitysave/Read/ReadVariableOp*
_output_shapes
:*
T0
^
save/Identity_1Identitysave/Identity"/device:CPU:0*
_output_shapes
:*
T0
�
save/Read_1/ReadVariableOpReadVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
dtype0*
_output_shapes
:
\
save/Identity_2Identitysave/Read_1/ReadVariableOp*
T0*
_output_shapes
:
`
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
_output_shapes
:*
T0
t
save/Read_2/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
\
save/Identity_4Identitysave/Read_2/ReadVariableOp*
T0*
_output_shapes
:
`
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
_output_shapes
:*
T0
{
save/Read_3/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	�*
dtype0
a
save/Identity_6Identitysave/Read_3/ReadVariableOp*
T0*
_output_shapes
:	�
e
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
_output_shapes
:	�*
T0
�
save/Read_4/ReadVariableOpReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0*
_output_shapes
:
\
save/Identity_8Identitysave/Read_4/ReadVariableOp*
T0*
_output_shapes
:
`
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
_output_shapes
:*
T0
�
save/Read_5/ReadVariableOpReadVariableOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
dtype0*
_output_shapes
:
]
save/Identity_10Identitysave/Read_5/ReadVariableOp*
T0*
_output_shapes
:
b
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes
:
t
save/Read_6/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
]
save/Identity_12Identitysave/Read_6/ReadVariableOp*
T0*
_output_shapes
:
b
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
T0*
_output_shapes
:
z
save/Read_7/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:*
dtype0
a
save/Identity_14Identitysave/Read_7/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
_output_shapes

:*
T0
m
save/Read_8/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
]
save/Identity_16Identitysave/Read_8/ReadVariableOp*
T0*
_output_shapes
:
b
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
_output_shapes
:*
T0
s
save/Read_9/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
_output_shapes

:*
dtype0
a
save/Identity_18Identitysave/Read_9/ReadVariableOp*
_output_shapes

:*
T0
f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
_output_shapes

:*
T0
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_834f4d372f344637b32d69c73a998e3f/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*�
value�B�B)dnn/hiddenlayer_0/batchnorm_0/moving_meanB-dnn/hiddenlayer_0/batchnorm_0/moving_varianceB)dnn/hiddenlayer_1/batchnorm_1/moving_meanB-dnn/hiddenlayer_1/batchnorm_1/moving_varianceBQdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weightsBQdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weightsBglobal_step*
dtype0
�
save/SaveV2/shape_and_slicesConst"/device:CPU:0*�
value|BzB B B B B126 25 0,126:0,25B35 19 0,35:0,19B168 25 0,168:0,25B4 5 0,4:0,5B29 19 0,29:0,19B247 25 0,247:0,25B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices=dnn/hiddenlayer_0/batchnorm_0/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_0/batchnorm_0/moving_variance/Read/ReadVariableOp=dnn/hiddenlayer_1/batchnorm_1/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_1/batchnorm_1/moving_variance/Read/ReadVariableOp]dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/read]dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/read\dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/read\dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/read\dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/read\dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/readglobal_step"/device:CPU:0*
dtypes
2	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :
�
save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/Read_10/ReadVariableOpReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_20Identitysave/Read_10/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_21Identitysave/Identity_20"/device:CPU:0*
T0*
_output_shapes
:
�
save/Read_11/ReadVariableOpReadVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_22Identitysave/Read_11/ReadVariableOp"/device:CPU:0*
_output_shapes
:*
T0
b
save/Identity_23Identitysave/Identity_22"/device:CPU:0*
_output_shapes
:*
T0
�
save/Read_12/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0"/device:CPU:0*
_output_shapes
:*
dtype0
m
save/Identity_24Identitysave/Read_12/ReadVariableOp"/device:CPU:0*
_output_shapes
:*
T0
b
save/Identity_25Identitysave/Identity_24"/device:CPU:0*
T0*
_output_shapes
:
�
save/Read_13/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes
:	�
r
save/Identity_26Identitysave/Read_13/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	�
g
save/Identity_27Identitysave/Identity_26"/device:CPU:0*
T0*
_output_shapes
:	�
�
save/Read_14/ReadVariableOpReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_28Identitysave/Read_14/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_29Identitysave/Identity_28"/device:CPU:0*
_output_shapes
:*
T0
�
save/Read_15/ReadVariableOpReadVariableOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_30Identitysave/Read_15/ReadVariableOp"/device:CPU:0*
_output_shapes
:*
T0
b
save/Identity_31Identitysave/Identity_30"/device:CPU:0*
_output_shapes
:*
T0
�
save/Read_16/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_32Identitysave/Read_16/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_33Identitysave/Identity_32"/device:CPU:0*
T0*
_output_shapes
:
�
save/Read_17/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_34Identitysave/Read_17/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_35Identitysave/Identity_34"/device:CPU:0*
_output_shapes

:*
T0
}
save/Read_18/ReadVariableOpReadVariableOpdnn/logits/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_36Identitysave/Read_18/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_37Identitysave/Identity_36"/device:CPU:0*
T0*
_output_shapes
:
�
save/Read_19/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_38Identitysave/Read_19/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
f
save/Identity_39Identitysave/Identity_38"/device:CPU:0*
T0*
_output_shapes

:
�
save/SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*�
value�B�
B"dnn/hiddenlayer_0/batchnorm_0/betaB#dnn/hiddenlayer_0/batchnorm_0/gammaBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelB"dnn/hiddenlayer_1/batchnorm_1/betaB#dnn/hiddenlayer_1/batchnorm_1/gammaBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernel*
dtype0
�
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*�
valuezBx
B16 0,16B16 0,16B16 0,16B129 16 0,129:0,16B16 0,16B16 0,16B16 0,16B16 16 0,16:0,16B1 0,1B16 1 0,16:0,1*
dtype0
�
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_21save/Identity_23save/Identity_25save/Identity_27save/Identity_29save/Identity_31save/Identity_33save/Identity_35save/Identity_37save/Identity_39"/device:CPU:0*
dtypes
2

�
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*
N*
_output_shapes
:*
T0
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
�
save/Identity_40Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�B)dnn/hiddenlayer_0/batchnorm_0/moving_meanB-dnn/hiddenlayer_0/batchnorm_0/moving_varianceB)dnn/hiddenlayer_1/batchnorm_1/moving_meanB-dnn/hiddenlayer_1/batchnorm_1/moving_varianceBQdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weightsBQdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weightsBglobal_step*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*�
value|BzB B B B B126 25 0,126:0,25B35 19 0,35:0,19B168 25 0,168:0,25B4 5 0,4:0,5B29 19 0,29:0,19B247 25 0,247:0,25B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*f
_output_shapesT
R:::::~:#:	�:::	�:*
dtypes
2	
O
save/Identity_41Identitysave/RestoreV2*
T0*
_output_shapes
:
s
save/AssignVariableOpAssignVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_meansave/Identity_41*
dtype0
Q
save/Identity_42Identitysave/RestoreV2:1*
T0*
_output_shapes
:
y
save/AssignVariableOp_1AssignVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variancesave/Identity_42*
dtype0
Q
save/Identity_43Identitysave/RestoreV2:2*
T0*
_output_shapes
:
u
save/AssignVariableOp_2AssignVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_meansave/Identity_43*
dtype0
Q
save/Identity_44Identitysave/RestoreV2:3*
_output_shapes
:*
T0
y
save/AssignVariableOp_3AssignVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variancesave/Identity_44*
dtype0
�
save/AssignAssignXdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0save/RestoreV2:4*
_output_shapes

:~*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0
�
save/Assign_1AssignXdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0save/RestoreV2:5*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
_output_shapes

:#
�
save/Assign_2AssignWdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0save/RestoreV2:6*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	�
�
save/Assign_3AssignWdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0save/RestoreV2:7*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
_output_shapes

:
�
save/Assign_4AssignWdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0save/RestoreV2:8*
_output_shapes

:*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0
�
save/Assign_5AssignWdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0save/RestoreV2:9*
_output_shapes
:	�*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0
x
save/Assign_6Assignglobal_stepsave/RestoreV2:10*
T0	*
_class
loc:@global_step*
_output_shapes
: 
�
save/restore_shardNoOp^save/Assign^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6
�
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*�
value�B�
B"dnn/hiddenlayer_0/batchnorm_0/betaB#dnn/hiddenlayer_0/batchnorm_0/gammaBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelB"dnn/hiddenlayer_1/batchnorm_1/betaB#dnn/hiddenlayer_1/batchnorm_1/gammaBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernel*
dtype0*
_output_shapes
:

�
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*�
valuezBx
B16 0,16B16 0,16B16 0,16B129 16 0,129:0,16B16 0,16B16 0,16B16 0,16B16 16 0,16:0,16B1 0,1B16 1 0,16:0,1*
dtype0*
_output_shapes
:

�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*]
_output_shapesK
I::::	�::::::*
dtypes
2

b
save/Identity_45Identitysave/RestoreV2_1"/device:CPU:0*
_output_shapes
:*
T0
�
save/AssignVariableOp_4AssignVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0save/Identity_45"/device:CPU:0*
dtype0
d
save/Identity_46Identitysave/RestoreV2_1:1"/device:CPU:0*
T0*
_output_shapes
:
�
save/AssignVariableOp_5AssignVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0save/Identity_46"/device:CPU:0*
dtype0
d
save/Identity_47Identitysave/RestoreV2_1:2"/device:CPU:0*
_output_shapes
:*
T0
x
save/AssignVariableOp_6AssignVariableOpdnn/hiddenlayer_0/bias/part_0save/Identity_47"/device:CPU:0*
dtype0
i
save/Identity_48Identitysave/RestoreV2_1:3"/device:CPU:0*
T0*
_output_shapes
:	�
z
save/AssignVariableOp_7AssignVariableOpdnn/hiddenlayer_0/kernel/part_0save/Identity_48"/device:CPU:0*
dtype0
d
save/Identity_49Identitysave/RestoreV2_1:4"/device:CPU:0*
_output_shapes
:*
T0
�
save/AssignVariableOp_8AssignVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0save/Identity_49"/device:CPU:0*
dtype0
d
save/Identity_50Identitysave/RestoreV2_1:5"/device:CPU:0*
_output_shapes
:*
T0
�
save/AssignVariableOp_9AssignVariableOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0save/Identity_50"/device:CPU:0*
dtype0
d
save/Identity_51Identitysave/RestoreV2_1:6"/device:CPU:0*
_output_shapes
:*
T0
y
save/AssignVariableOp_10AssignVariableOpdnn/hiddenlayer_1/bias/part_0save/Identity_51"/device:CPU:0*
dtype0
h
save/Identity_52Identitysave/RestoreV2_1:7"/device:CPU:0*
T0*
_output_shapes

:
{
save/AssignVariableOp_11AssignVariableOpdnn/hiddenlayer_1/kernel/part_0save/Identity_52"/device:CPU:0*
dtype0
d
save/Identity_53Identitysave/RestoreV2_1:8"/device:CPU:0*
T0*
_output_shapes
:
r
save/AssignVariableOp_12AssignVariableOpdnn/logits/bias/part_0save/Identity_53"/device:CPU:0*
dtype0
h
save/Identity_54Identitysave/RestoreV2_1:9"/device:CPU:0*
_output_shapes

:*
T0
t
save/AssignVariableOp_13AssignVariableOpdnn/logits/kernel/part_0save/Identity_54"/device:CPU:0*
dtype0
�
save/restore_shard_1NoOp^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"?
save/Const:0save/Identity_40:0save/restore_all (5 @F8"�*
trainable_variables�*�*
�
Zdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights~  "~2wdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
Zdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights#  "#2wdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
Ydnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/read:0"`
Pdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights�  "�2vdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
Ydnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/read:0"^
Pdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights  "2vdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
Ydnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/read:0"^
Pdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights  "2vdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
Ydnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/read:0"`
Pdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights�  "�2vdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"(
dnn/hiddenlayer_0/kernel�  "�(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
�
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias "(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
�
,dnn/hiddenlayer_0/batchnorm_0/gamma/part_0:01dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Assign@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Read/ReadVariableOp:0".
#dnn/hiddenlayer_0/batchnorm_0/gamma "(2=dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Initializer/ones:08
�
+dnn/hiddenlayer_0/batchnorm_0/beta/part_0:00dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Assign?dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Read/ReadVariableOp:0"-
"dnn/hiddenlayer_0/batchnorm_0/beta "(2=dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Initializer/zeros:08
�
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel  "(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
�
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias "(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
�
,dnn/hiddenlayer_1/batchnorm_1/gamma/part_0:01dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Assign@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Read/ReadVariableOp:0".
#dnn/hiddenlayer_1/batchnorm_1/gamma "(2=dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Initializer/ones:08
�
+dnn/hiddenlayer_1/batchnorm_1/beta/part_0:00dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Assign?dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Read/ReadVariableOp:0"-
"dnn/hiddenlayer_1/batchnorm_1/beta "(2=dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Initializer/zeros:08
�
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel  "(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
�
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08"�
	summaries�
�
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"�2
	variables�2�2
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
�
Zdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights~  "~2wdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
Zdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights#  "#2wdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
Ydnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/read:0"`
Pdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights�  "�2vdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
Ydnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/read:0"^
Pdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights  "2vdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
Ydnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/read:0"^
Pdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights  "2vdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
Ydnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/read:0"`
Pdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights�  "�2vdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
�
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"(
dnn/hiddenlayer_0/kernel�  "�(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
�
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias "(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
�
,dnn/hiddenlayer_0/batchnorm_0/gamma/part_0:01dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Assign@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Read/ReadVariableOp:0".
#dnn/hiddenlayer_0/batchnorm_0/gamma "(2=dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Initializer/ones:08
�
+dnn/hiddenlayer_0/batchnorm_0/beta/part_0:00dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Assign?dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Read/ReadVariableOp:0"-
"dnn/hiddenlayer_0/batchnorm_0/beta "(2=dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Initializer/zeros:08
�
+dnn/hiddenlayer_0/batchnorm_0/moving_mean:00dnn/hiddenlayer_0/batchnorm_0/moving_mean/Assign?dnn/hiddenlayer_0/batchnorm_0/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_0/batchnorm_0/moving_mean/Initializer/zeros:0
�
/dnn/hiddenlayer_0/batchnorm_0/moving_variance:04dnn/hiddenlayer_0/batchnorm_0/moving_variance/AssignCdnn/hiddenlayer_0/batchnorm_0/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_0/batchnorm_0/moving_variance/Initializer/ones:0
�
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel  "(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
�
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias "(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
�
,dnn/hiddenlayer_1/batchnorm_1/gamma/part_0:01dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Assign@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Read/ReadVariableOp:0".
#dnn/hiddenlayer_1/batchnorm_1/gamma "(2=dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Initializer/ones:08
�
+dnn/hiddenlayer_1/batchnorm_1/beta/part_0:00dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Assign?dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Read/ReadVariableOp:0"-
"dnn/hiddenlayer_1/batchnorm_1/beta "(2=dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Initializer/zeros:08
�
+dnn/hiddenlayer_1/batchnorm_1/moving_mean:00dnn/hiddenlayer_1/batchnorm_1/moving_mean/Assign?dnn/hiddenlayer_1/batchnorm_1/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_1/batchnorm_1/moving_mean/Initializer/zeros:0
�
/dnn/hiddenlayer_1/batchnorm_1/moving_variance:04dnn/hiddenlayer_1/batchnorm_1/moving_variance/AssignCdnn/hiddenlayer_1/batchnorm_1/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_1/batchnorm_1/moving_variance/Initializer/ones:0
�
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel  "(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
�
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"�"
cond_context�"�"
�
 dnn/zero_fraction/cond/cond_text dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_t:0 *�
/dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1:0
dnn/zero_fraction/cond/Cast:0
+dnn/zero_fraction/cond/count_nonzero/Cast:0
,dnn/zero_fraction/cond/count_nonzero/Const:0
6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
/dnn/zero_fraction/cond/count_nonzero/NotEqual:0
4dnn/zero_fraction/cond/count_nonzero/nonzero_count:0
,dnn/zero_fraction/cond/count_nonzero/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_t:0i
/dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1:06dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0
�
"dnn/zero_fraction/cond/cond_text_1 dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_f:0*�
/dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1:0
-dnn/zero_fraction/cond/count_nonzero_1/Cast:0
.dnn/zero_fraction/cond/count_nonzero_1/Const:0
8dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
1dnn/zero_fraction/cond/count_nonzero_1/NotEqual:0
6dnn/zero_fraction/cond/count_nonzero_1/nonzero_count:0
.dnn/zero_fraction/cond/count_nonzero_1/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_f:0k
/dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1:08dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0
�
"dnn/zero_fraction_1/cond/cond_text"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_t:0 *�
/dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1:0
dnn/zero_fraction_1/cond/Cast:0
-dnn/zero_fraction_1/cond/count_nonzero/Cast:0
.dnn/zero_fraction_1/cond/count_nonzero/Const:0
8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_1/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_1/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_1/cond/count_nonzero/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_t:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0k
/dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1:08dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
�
$dnn/zero_fraction_1/cond/cond_text_1"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_f:0*�
/dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1:0
/dnn/zero_fraction_1/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_1/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_1/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_f:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0m
/dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1:0:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
�
"dnn/zero_fraction_2/cond/cond_text"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_t:0 *�
dnn/logits/BiasAdd:0
dnn/zero_fraction_2/cond/Cast:0
-dnn/zero_fraction_2/cond/count_nonzero/Cast:0
.dnn/zero_fraction_2/cond/count_nonzero/Const:0
8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_2/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_2/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_2/cond/count_nonzero/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_t:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0P
dnn/logits/BiasAdd:08dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
�
$dnn/zero_fraction_2/cond/cond_text_1"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_f:0*�
dnn/logits/BiasAdd:0
/dnn/zero_fraction_2/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_2/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_2/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_f:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0R
dnn/logits/BiasAdd:0:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0"%
saved_model_main_op


group_deps*�

regression�
3
inputs)
input_example_tensor:0���������6
outputs+
dnn/logits/BiasAdd:0���������tensorflow/serving/regress*�
serving_default�
3
inputs)
input_example_tensor:0���������6
outputs+
dnn/logits/BiasAdd:0���������tensorflow/serving/regress*�
predict�
5
examples)
input_example_tensor:0���������:
predictions+
dnn/logits/BiasAdd:0���������tensorflow/serving/predict