Љу5
с*С*
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
B
Equal
x"T
y"T
z
"
Ttype:
2	


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
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
я
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

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
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
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
list(type)(0
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
З
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
і
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

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

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
shapeshape
9
VarIsInitializedOp
resource
is_initialized

s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
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
b'unknown'8Жр)
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
shape: *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
o
input_example_tensorPlaceholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
U
ParseExample/ConstConst*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_1Const*
_output_shapes
: *
valueB *
dtype0
W
ParseExample/Const_2Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_3Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_4Const*
_output_shapes
: *
valueB *
dtype0
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
ParseExample/ParseExample/namesConst*
dtype0*
_output_shapes
: *
valueB 
r
'ParseExample/ParseExample/sparse_keys_0Const*
valueB B
feature_14*
dtype0*
_output_shapes
: 
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
'ParseExample/ParseExample/sparse_keys_3Const*
valueB B	feature_7*
dtype0*
_output_shapes
: 
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
&ParseExample/ParseExample/dense_keys_0Const*
_output_shapes
: *
valueB B
feature_10*
dtype0
q
&ParseExample/ParseExample/dense_keys_1Const*
valueB B
feature_11*
dtype0*
_output_shapes
: 
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
&ParseExample/ParseExample/dense_keys_7Const*
valueB B	feature_3*
dtype0*
_output_shapes
: 
p
&ParseExample/ParseExample/dense_keys_8Const*
valueB B	feature_4*
dtype0*
_output_shapes
: 
p
&ParseExample/ParseExample/dense_keys_9Const*
dtype0*
_output_shapes
: *
valueB B	feature_5
q
'ParseExample/ParseExample/dense_keys_10Const*
valueB B	feature_6*
dtype0*
_output_shapes
: 

ParseExample/ParseExampleParseExampleinput_example_tensorParseExample/ParseExample/names'ParseExample/ParseExample/sparse_keys_0'ParseExample/ParseExample/sparse_keys_1'ParseExample/ParseExample/sparse_keys_2'ParseExample/ParseExample/sparse_keys_3'ParseExample/ParseExample/sparse_keys_4'ParseExample/ParseExample/sparse_keys_5&ParseExample/ParseExample/dense_keys_0&ParseExample/ParseExample/dense_keys_1&ParseExample/ParseExample/dense_keys_2&ParseExample/ParseExample/dense_keys_3&ParseExample/ParseExample/dense_keys_4&ParseExample/ParseExample/dense_keys_5&ParseExample/ParseExample/dense_keys_6&ParseExample/ParseExample/dense_keys_7&ParseExample/ParseExample/dense_keys_8&ParseExample/ParseExample/dense_keys_9'ParseExample/ParseExample/dense_keys_10ParseExample/ConstParseExample/Const_1ParseExample/Const_2ParseExample/Const_3ParseExample/Const_4ParseExample/Const_5ParseExample/Const_6ParseExample/Const_7ParseExample/Const_8ParseExample/Const_9ParseExample/Const_10*
Tdense
2*
Ndense*з
_output_shapesФ
С:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
Nsparse*
sparse_types

2*T
dense_shapesD
B:::::::::::
c
ExponentialDecay/learning_rateConst*
valueB
 *џцл.*
dtype0*
_output_shapes
: 
Z
ExponentialDecay/Cast/xConst*
value
B :'*
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
 *Тu?*
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
ExponentialDecay/PowPowExponentialDecay/Cast_1/xExponentialDecay/truediv*
T0*
_output_shapes
: 
n
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
T0*
_output_shapes
: 
Й
{dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"m      *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
Ќ
zdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Ў
|dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *ЭЬL>*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
dtype0
в
dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
dtype0*
_output_shapes

:m*

seed**
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
seed2.

ydnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuldnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal|dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes

:m*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0

udnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normalAddydnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulzdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
_output_shapes

:m

Xdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0
VariableV2*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:m*
shape
:m
а
_dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/AssignAssignXdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0udnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal*
_output_shapes

:m*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0
й
]dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/readIdentityXdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
_output_shapes

:m*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0
Й
{dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"       *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
dtype0
Ќ
zdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Ў
|dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *ѕыj>*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
в
dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*

seed**
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
seed27*
dtype0*
_output_shapes

: 

ydnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuldnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal|dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes

: *
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0

udnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normalAddydnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulzdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
_output_shapes

: 

Xdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0
VariableV2*
shape
: *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

: 
а
_dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/AssignAssignXdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0udnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
_output_shapes

: 
й
]dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/readIdentityXdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
_output_shapes

: 
З
zdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"      *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
Њ
ydnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Ќ
{dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0
а
dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalzdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
_output_shapes
:	*

seed**
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
seed2@*
dtype0

xdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuldnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes
:	*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0

tdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normalAddxdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulydnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
_output_shapes
:	*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0

Wdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0
VariableV2*
_output_shapes
:	*
shape:	*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
dtype0
Э
^dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/AssignAssignWdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0tdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	
з
\dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/readIdentityWdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0
З
zdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"      *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
Њ
ydnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0
Ќ
{dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *.љф>*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
dtype0
Я
dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalzdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
dtype0*
_output_shapes

:*

seed**
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
seed2I

xdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuldnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes

:*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0

tdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normalAddxdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulydnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
_output_shapes

:

Wdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0
VariableV2*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:*
shape
:
Ь
^dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/AssignAssignWdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0tdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal*
_output_shapes

:*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0
ж
\dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/readIdentityWdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
_output_shapes

:
З
zdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"      *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
Њ
ydnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Ќ
{dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *ѕыj>*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Я
dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalzdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*

seed**
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
seed2R*
dtype0*
_output_shapes

:

xdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuldnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
_output_shapes

:

tdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normalAddxdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulydnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
_output_shapes

:

Wdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0
VariableV2*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:*
shape
:
Ь
^dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/AssignAssignWdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0tdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal*
_output_shapes

:*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0
ж
\dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/readIdentityWdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
_output_shapes

:*
T0
З
zdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"Ѕ      *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0
Њ
ydnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Ќ
{dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *ЭЬL>*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
а
dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalzdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:	Ѕ*

seed**
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
seed2[

xdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuldnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal{dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes
:	Ѕ*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0

tdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normalAddxdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulydnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
_output_shapes
:	Ѕ*
T0

Wdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0
VariableV2*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:	Ѕ*
shape:	Ѕ
Э
^dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/AssignAssignWdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0tdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
_output_shapes
:	Ѕ
з
\dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/readIdentityWdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
_output_shapes
:	Ѕ

;dnn/input_from_feature_columns/input_layer/feature_10/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *-N
Э
9dnn/input_from_feature_columns/input_layer/feature_10/subSubParseExample/ParseExample:18;dnn/input_from_feature_columns/input_layer/feature_10/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

?dnn/input_from_feature_columns/input_layer/feature_10/truediv/yConst*
dtype0*
_output_shapes
: *
valueB
 *ЎАN
і
=dnn/input_from_feature_columns/input_layer/feature_10/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_10/sub?dnn/input_from_feature_columns/input_layer/feature_10/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ј
;dnn/input_from_feature_columns/input_layer/feature_10/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_10/truediv*
_output_shapes
:*
T0

Idnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Л
Cdnn/input_from_feature_columns/input_layer/feature_10/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_10/ShapeIdnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_10/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 

Ednn/input_from_feature_columns/input_layer/feature_10/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Cdnn/input_from_feature_columns/input_layer/feature_10/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_10/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_10/Reshape/shape/1*
T0*
N*
_output_shapes
:
ў
=dnn/input_from_feature_columns/input_layer/feature_10/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_10/truedivCdnn/input_from_feature_columns/input_layer/feature_10/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0

;dnn/input_from_feature_columns/input_layer/feature_11/sub/yConst*
valueB
 *Мц?*
dtype0*
_output_shapes
: 
Э
9dnn/input_from_feature_columns/input_layer/feature_11/subSubParseExample/ParseExample:19;dnn/input_from_feature_columns/input_layer/feature_11/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

?dnn/input_from_feature_columns/input_layer/feature_11/truediv/yConst*
valueB
 * д <*
dtype0*
_output_shapes
: 
і
=dnn/input_from_feature_columns/input_layer/feature_11/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_11/sub?dnn/input_from_feature_columns/input_layer/feature_11/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ј
;dnn/input_from_feature_columns/input_layer/feature_11/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_11/truediv*
_output_shapes
:*
T0

Idnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Л
Cdnn/input_from_feature_columns/input_layer/feature_11/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_11/ShapeIdnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_11/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Ednn/input_from_feature_columns/input_layer/feature_11/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Cdnn/input_from_feature_columns/input_layer/feature_11/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_11/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_11/Reshape/shape/1*
T0*
N*
_output_shapes
:
ў
=dnn/input_from_feature_columns/input_layer/feature_11/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_11/truedivCdnn/input_from_feature_columns/input_layer/feature_11/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0

;dnn/input_from_feature_columns/input_layer/feature_12/sub/yConst*
valueB
 *ђ№@*
dtype0*
_output_shapes
: 
Э
9dnn/input_from_feature_columns/input_layer/feature_12/subSubParseExample/ParseExample:20;dnn/input_from_feature_columns/input_layer/feature_12/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

?dnn/input_from_feature_columns/input_layer/feature_12/truediv/yConst*
valueB
 *Ч_B*
dtype0*
_output_shapes
: 
і
=dnn/input_from_feature_columns/input_layer/feature_12/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_12/sub?dnn/input_from_feature_columns/input_layer/feature_12/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ј
;dnn/input_from_feature_columns/input_layer/feature_12/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_12/truediv*
T0*
_output_shapes
:

Idnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Л
Cdnn/input_from_feature_columns/input_layer/feature_12/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_12/ShapeIdnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_12/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ednn/input_from_feature_columns/input_layer/feature_12/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0

Cdnn/input_from_feature_columns/input_layer/feature_12/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_12/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_12/Reshape/shape/1*
T0*
N*
_output_shapes
:
ў
=dnn/input_from_feature_columns/input_layer/feature_12/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_12/truedivCdnn/input_from_feature_columns/input_layer/feature_12/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0

;dnn/input_from_feature_columns/input_layer/feature_13/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *ЯШDD
Э
9dnn/input_from_feature_columns/input_layer/feature_13/subSubParseExample/ParseExample:21;dnn/input_from_feature_columns/input_layer/feature_13/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

?dnn/input_from_feature_columns/input_layer/feature_13/truediv/yConst*
valueB
 *hЬ\D*
dtype0*
_output_shapes
: 
і
=dnn/input_from_feature_columns/input_layer/feature_13/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_13/sub?dnn/input_from_feature_columns/input_layer/feature_13/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ј
;dnn/input_from_feature_columns/input_layer/feature_13/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_13/truediv*
_output_shapes
:*
T0

Idnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Kdnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Л
Cdnn/input_from_feature_columns/input_layer/feature_13/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_13/ShapeIdnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_13/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0

Ednn/input_from_feature_columns/input_layer/feature_13/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Cdnn/input_from_feature_columns/input_layer/feature_13/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_13/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_13/Reshape/shape/1*
_output_shapes
:*
T0*
N
ў
=dnn/input_from_feature_columns/input_layer/feature_13/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_13/truedivCdnn/input_from_feature_columns/input_layer/feature_13/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Е
Fdnn/input_from_feature_columns/input_layer/feature_14_embedding/lookupStringToHashBucketFastParseExample/ParseExample:6*#
_output_shapes
:џџџџџџџџџ*
num_bucketsm
В
hdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
Б
gdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SliceSliceParseExample/ParseExample:12hdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice/begingdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
Ќ
bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
в
adnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/ProdProdbdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slicebdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Const*
_output_shapes
: *
T0	
Џ
mdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
Ќ
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ў
ednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:12mdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2/indicesjdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2/axis*
_output_shapes
: *
Taxis0*
Tindices0*
Tparams0	
у
cdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Cast/xPackadnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Prodednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Ш
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExampleParseExample/ParseExample:12cdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ѕ
sdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshape/IdentityIdentityFdnn/input_from_feature_columns/input_layer/feature_14_embedding/lookup*
T0	*#
_output_shapes
:џџџџџџџџџ
­
kdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 

idnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GreaterEqualGreaterEqualsdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshape/Identitykdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:џџџџџџџџџ
џ
bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/WhereWhereidnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
Н
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
э
ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/ReshapeReshapebdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Wherejdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:џџџџџџџџџ
Ў
ldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

gdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_1GatherV2jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshapeddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_1/axis*
Tindices0	*
Tparams0	*'
_output_shapes
:џџџџџџџџџ*
Taxis0
Ў
ldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

gdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_2GatherV2sdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshape/Identityddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_2/axis*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	

ednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/IdentityIdentityldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
И
vdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Ќ
dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsgdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_1gdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/GatherV2_2ednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Identityvdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
к
dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
м
dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
м
dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
Д
dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stackdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0	
У
ydnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/CastCastdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
Ы
{dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/UniqueUniquednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
К
dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Н
dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2]dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/read{dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/Uniquednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0
е
dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitydnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
tdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparseSparseSegmentMeandnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity}dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/Unique:1ydnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
Н
ldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
valueB"џџџџ   *
dtype0

fdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_1Reshapednn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ldnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ

bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/ShapeShapetdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
К
pdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
М
rdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
М
rdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
ў
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_sliceStridedSlicebdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Shapepdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stackrdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_1rdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
І
ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
ъ
bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/stackPackddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/stack/0jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
№
adnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/TileTilefdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_1bdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/stack*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

gdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/zeros_like	ZerosLiketdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
к
\dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weightsSelectadnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Tilegdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/zeros_liketdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
Н
cdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Cast_1CastParseExample/ParseExample:12*
_output_shapes
:*

DstT0*

SrcT0	
Д
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
Г
idnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
л
ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1Slicecdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Cast_1jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1/beginidnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
№
ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Shape_1Shape\dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights*
T0*
_output_shapes
:
Д
jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
М
idnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
м
ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2Sliceddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Shape_1jdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2/beginidnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Њ
hdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
г
cdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/concatConcatV2ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_1ddnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Slice_2hdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
ц
fdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_2Reshape\dnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weightscdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/concat*
T0*'
_output_shapes
:џџџџџџџџџ
л
Ednn/input_from_feature_columns/input_layer/feature_14_embedding/ShapeShapefdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_2*
_output_shapes
:*
T0

Sdnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0

Udnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Udnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
э
Mdnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_sliceStridedSliceEdnn/input_from_feature_columns/input_layer/feature_14_embedding/ShapeSdnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stackUdnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stack_1Udnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Odnn/input_from_feature_columns/input_layer/feature_14_embedding/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
Ѓ
Mdnn/input_from_feature_columns/input_layer/feature_14_embedding/Reshape/shapePackMdnn/input_from_feature_columns/input_layer/feature_14_embedding/strided_sliceOdnn/input_from_feature_columns/input_layer/feature_14_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Л
Gdnn/input_from_feature_columns/input_layer/feature_14_embedding/ReshapeReshapefdnn/input_from_feature_columns/input_layer/feature_14_embedding/feature_14_embedding_weights/Reshape_2Mdnn/input_from_feature_columns/input_layer/feature_14_embedding/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ

;dnn/input_from_feature_columns/input_layer/feature_15/sub/yConst*
valueB
 *c@*
dtype0*
_output_shapes
: 
Э
9dnn/input_from_feature_columns/input_layer/feature_15/subSubParseExample/ParseExample:22;dnn/input_from_feature_columns/input_layer/feature_15/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

?dnn/input_from_feature_columns/input_layer/feature_15/truediv/yConst*
valueB
 *[8A*
dtype0*
_output_shapes
: 
і
=dnn/input_from_feature_columns/input_layer/feature_15/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_15/sub?dnn/input_from_feature_columns/input_layer/feature_15/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ј
;dnn/input_from_feature_columns/input_layer/feature_15/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_15/truediv*
T0*
_output_shapes
:

Idnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Kdnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Л
Cdnn/input_from_feature_columns/input_layer/feature_15/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_15/ShapeIdnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_15/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Ednn/input_from_feature_columns/input_layer/feature_15/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Cdnn/input_from_feature_columns/input_layer/feature_15/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_15/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_15/Reshape/shape/1*
T0*
N*
_output_shapes
:
ў
=dnn/input_from_feature_columns/input_layer/feature_15/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_15/truedivCdnn/input_from_feature_columns/input_layer/feature_15/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ

;dnn/input_from_feature_columns/input_layer/feature_16/sub/yConst*
valueB
 *EрA*
dtype0*
_output_shapes
: 
Э
9dnn/input_from_feature_columns/input_layer/feature_16/subSubParseExample/ParseExample:23;dnn/input_from_feature_columns/input_layer/feature_16/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

?dnn/input_from_feature_columns/input_layer/feature_16/truediv/yConst*
_output_shapes
: *
valueB
 *нгB*
dtype0
і
=dnn/input_from_feature_columns/input_layer/feature_16/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_16/sub?dnn/input_from_feature_columns/input_layer/feature_16/truediv/y*'
_output_shapes
:џџџџџџџџџ*
T0
Ј
;dnn/input_from_feature_columns/input_layer/feature_16/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_16/truediv*
T0*
_output_shapes
:

Idnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0

Kdnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Л
Cdnn/input_from_feature_columns/input_layer/feature_16/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_16/ShapeIdnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_16/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 

Ednn/input_from_feature_columns/input_layer/feature_16/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Cdnn/input_from_feature_columns/input_layer/feature_16/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_16/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_16/Reshape/shape/1*
_output_shapes
:*
T0*
N
ў
=dnn/input_from_feature_columns/input_layer/feature_16/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_16/truedivCdnn/input_from_feature_columns/input_layer/feature_16/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0

;dnn/input_from_feature_columns/input_layer/feature_17/sub/yConst*
valueB
 *ВSA*
dtype0*
_output_shapes
: 
Э
9dnn/input_from_feature_columns/input_layer/feature_17/subSubParseExample/ParseExample:24;dnn/input_from_feature_columns/input_layer/feature_17/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

?dnn/input_from_feature_columns/input_layer/feature_17/truediv/yConst*
_output_shapes
: *
valueB
 *дB*
dtype0
і
=dnn/input_from_feature_columns/input_layer/feature_17/truedivRealDiv9dnn/input_from_feature_columns/input_layer/feature_17/sub?dnn/input_from_feature_columns/input_layer/feature_17/truediv/y*'
_output_shapes
:џџџџџџџџџ*
T0
Ј
;dnn/input_from_feature_columns/input_layer/feature_17/ShapeShape=dnn/input_from_feature_columns/input_layer/feature_17/truediv*
T0*
_output_shapes
:

Idnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Kdnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Kdnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Л
Cdnn/input_from_feature_columns/input_layer/feature_17/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/feature_17/ShapeIdnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stackKdnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/feature_17/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 

Ednn/input_from_feature_columns/input_layer/feature_17/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Cdnn/input_from_feature_columns/input_layer/feature_17/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/feature_17/strided_sliceEdnn/input_from_feature_columns/input_layer/feature_17/Reshape/shape/1*
N*
_output_shapes
:*
T0
ў
=dnn/input_from_feature_columns/input_layer/feature_17/ReshapeReshape=dnn/input_from_feature_columns/input_layer/feature_17/truedivCdnn/input_from_feature_columns/input_layer/feature_17/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0
Е
Fdnn/input_from_feature_columns/input_layer/feature_18_embedding/lookupStringToHashBucketFastParseExample/ParseExample:7*#
_output_shapes
:џџџџџџџџџ*
num_buckets 
В
hdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
Б
gdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SliceSliceParseExample/ParseExample:13hdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice/begingdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
Ќ
bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
в
adnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/ProdProdbdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slicebdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Const*
T0	*
_output_shapes
: 
Џ
mdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
Ќ
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ў
ednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:13mdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2/indicesjdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
у
cdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Cast/xPackadnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Prodednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Ъ
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:1ParseExample/ParseExample:13cdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ѕ
sdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshape/IdentityIdentityFdnn/input_from_feature_columns/input_layer/feature_18_embedding/lookup*#
_output_shapes
:џџџџџџџџџ*
T0	
­
kdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	

idnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GreaterEqualGreaterEqualsdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshape/Identitykdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:џџџџџџџџџ
џ
bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/WhereWhereidnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
Н
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
э
ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/ReshapeReshapebdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Wherejdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:џџџџџџџџџ
Ў
ldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

gdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_1GatherV2jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshapeddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_1/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	
Ў
ldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

gdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_2GatherV2sdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshape/Identityddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:џџџџџџџџџ*
Taxis0

ednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/IdentityIdentityldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
И
vdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Ќ
dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsgdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_1gdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/GatherV2_2ednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Identityvdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
к
dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
м
dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
м
dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Д
dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stackdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ*
T0	*
Index0
У
ydnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/CastCastdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
Ы
{dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/UniqueUniquednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
К
dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Н
dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2]dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/read{dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/Uniquednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*
Tindices0	*
Tparams0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*'
_output_shapes
:џџџџџџџџџ
е
dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitydnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:џџџџџџџџџ*
T0
Ц
tdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparseSparseSegmentMeandnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity}dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/Unique:1ydnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
Н
ldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

fdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_1Reshapednn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ldnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ

bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/ShapeShapetdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
К
pdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
М
rdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
М
rdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ў
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_sliceStridedSlicebdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Shapepdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stackrdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_1rdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
І
ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
ъ
bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/stackPackddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/stack/0jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
№
adnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/TileTilefdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_1bdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/stack*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

gdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/zeros_like	ZerosLiketdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
к
\dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weightsSelectadnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Tilegdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/zeros_liketdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
Н
cdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Cast_1CastParseExample/ParseExample:13*
_output_shapes
:*

DstT0*

SrcT0	
Д
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
Г
idnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
л
ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1Slicecdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Cast_1jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1/beginidnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
№
ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Shape_1Shape\dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights*
_output_shapes
:*
T0
Д
jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
М
idnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
м
ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2Sliceddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Shape_1jdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2/beginidnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Њ
hdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
г
cdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/concatConcatV2ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_1ddnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Slice_2hdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
ц
fdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_2Reshape\dnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weightscdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/concat*
T0*'
_output_shapes
:џџџџџџџџџ
л
Ednn/input_from_feature_columns/input_layer/feature_18_embedding/ShapeShapefdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_2*
T0*
_output_shapes
:

Sdnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0

Udnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Udnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
э
Mdnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_sliceStridedSliceEdnn/input_from_feature_columns/input_layer/feature_18_embedding/ShapeSdnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stackUdnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stack_1Udnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Odnn/input_from_feature_columns/input_layer/feature_18_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
Ѓ
Mdnn/input_from_feature_columns/input_layer/feature_18_embedding/Reshape/shapePackMdnn/input_from_feature_columns/input_layer/feature_18_embedding/strided_sliceOdnn/input_from_feature_columns/input_layer/feature_18_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Л
Gdnn/input_from_feature_columns/input_layer/feature_18_embedding/ReshapeReshapefdnn/input_from_feature_columns/input_layer/feature_18_embedding/feature_18_embedding_weights/Reshape_2Mdnn/input_from_feature_columns/input_layer/feature_18_embedding/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Е
Ednn/input_from_feature_columns/input_layer/feature_2_embedding/lookupStringToHashBucketFastParseExample/ParseExample:8*#
_output_shapes
:џџџџџџџџџ*
num_buckets
А
fdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
Џ
ednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SliceSliceParseExample/ParseExample:14fdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice/beginednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
Њ
`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ь
_dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/ProdProd`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Const*
T0	*
_output_shapes
: 
­
kdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Њ
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ј
cdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:14kdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2/indiceshdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2/axis*
_output_shapes
: *
Taxis0*
Tindices0*
Tparams0	
н
adnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Cast/xPack_dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Ц
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:2ParseExample/ParseExample:14adnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ђ
qdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshape/IdentityIdentityEdnn/input_from_feature_columns/input_layer/feature_2_embedding/lookup*
T0	*#
_output_shapes
:џџџџџџџџџ
Ћ
idnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 

gdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GreaterEqualGreaterEqualqdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshape/Identityidnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:џџџџџџџџџ
ћ
`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/WhereWheregdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
Л
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
ч
bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/ReshapeReshape`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Wherehdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:џџџџџџџџџ
Ќ
jdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 

ednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_1GatherV2hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshapebdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_1/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	
Ќ
jdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

ednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_2GatherV2qdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshape/Identitybdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_2/axis*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	

cdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/IdentityIdentityjdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
Ж
tdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Ђ
dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_1ednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/GatherV2_2cdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Identitytdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
и
dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
к
dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
к
dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Њ
dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stackdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0	
П
wdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/CastCastdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
Ч
ydnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/UniqueUniquednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
З
dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Е
dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2\dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/readydnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/Uniquednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0
б
dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitydnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:џџџџџџџџџ*
T0
О
rdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparseSparseSegmentMeandnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity{dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/Unique:1wdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
Л
jdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

ddnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_1Reshapednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2jdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_1/shape*'
_output_shapes
:џџџџџџџџџ*
T0


`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/ShapeShaperdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
И
ndnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
К
pdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
К
pdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
є
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Shapendnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stackpdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_1pdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
Є
bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
ф
`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/stackPackbdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/stack/0hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/strided_slice*
N*
_output_shapes
:*
T0
ъ
_dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/TileTileddnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_1`dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/stack*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

ednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/zeros_like	ZerosLikerdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
в
Zdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weightsSelect_dnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Tileednn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/zeros_likerdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
Л
adnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Cast_1CastParseExample/ParseExample:14*

SrcT0	*
_output_shapes
:*

DstT0
В
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
Б
gdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
г
bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1Sliceadnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Cast_1hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1/begingdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
ь
bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Shape_1ShapeZdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights*
T0*
_output_shapes
:
В
hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
К
gdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
д
bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2Slicebdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Shape_1hdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2/begingdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ј
fdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ы
adnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/concatConcatV2bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_1bdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Slice_2fdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
р
ddnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_2ReshapeZdnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weightsadnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/concat*'
_output_shapes
:џџџџџџџџџ*
T0
и
Ddnn/input_from_feature_columns/input_layer/feature_2_embedding/ShapeShapeddnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_2*
T0*
_output_shapes
:

Rdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Tdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0

Tdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ш
Ldnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/feature_2_embedding/ShapeRdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stackTdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0

Ndnn/input_from_feature_columns/input_layer/feature_2_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
 
Ldnn/input_from_feature_columns/input_layer/feature_2_embedding/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/feature_2_embedding/strided_sliceNdnn/input_from_feature_columns/input_layer/feature_2_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
З
Fdnn/input_from_feature_columns/input_layer/feature_2_embedding/ReshapeReshapeddnn/input_from_feature_columns/input_layer/feature_2_embedding/feature_2_embedding_weights/Reshape_2Ldnn/input_from_feature_columns/input_layer/feature_2_embedding/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0

:dnn/input_from_feature_columns/input_layer/feature_3/sub/yConst*
valueB
 */QD*
dtype0*
_output_shapes
: 
Ы
8dnn/input_from_feature_columns/input_layer/feature_3/subSubParseExample/ParseExample:25:dnn/input_from_feature_columns/input_layer/feature_3/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

>dnn/input_from_feature_columns/input_layer/feature_3/truediv/yConst*
valueB
 *O ИD*
dtype0*
_output_shapes
: 
ѓ
<dnn/input_from_feature_columns/input_layer/feature_3/truedivRealDiv8dnn/input_from_feature_columns/input_layer/feature_3/sub>dnn/input_from_feature_columns/input_layer/feature_3/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
І
:dnn/input_from_feature_columns/input_layer/feature_3/ShapeShape<dnn/input_from_feature_columns/input_layer/feature_3/truediv*
_output_shapes
:*
T0

Hdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Jdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Jdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ж
Bdnn/input_from_feature_columns/input_layer/feature_3/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/feature_3/ShapeHdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stackJdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/feature_3/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 

Ddnn/input_from_feature_columns/input_layer/feature_3/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :

Bdnn/input_from_feature_columns/input_layer/feature_3/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/feature_3/strided_sliceDdnn/input_from_feature_columns/input_layer/feature_3/Reshape/shape/1*
N*
_output_shapes
:*
T0
ћ
<dnn/input_from_feature_columns/input_layer/feature_3/ReshapeReshape<dnn/input_from_feature_columns/input_layer/feature_3/truedivBdnn/input_from_feature_columns/input_layer/feature_3/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ

:dnn/input_from_feature_columns/input_layer/feature_4/sub/yConst*
valueB
 *dDB*
dtype0*
_output_shapes
: 
Ы
8dnn/input_from_feature_columns/input_layer/feature_4/subSubParseExample/ParseExample:26:dnn/input_from_feature_columns/input_layer/feature_4/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

>dnn/input_from_feature_columns/input_layer/feature_4/truediv/yConst*
dtype0*
_output_shapes
: *
valueB
 *§WпB
ѓ
<dnn/input_from_feature_columns/input_layer/feature_4/truedivRealDiv8dnn/input_from_feature_columns/input_layer/feature_4/sub>dnn/input_from_feature_columns/input_layer/feature_4/truediv/y*'
_output_shapes
:џџџџџџџџџ*
T0
І
:dnn/input_from_feature_columns/input_layer/feature_4/ShapeShape<dnn/input_from_feature_columns/input_layer/feature_4/truediv*
T0*
_output_shapes
:

Hdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Jdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Jdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ж
Bdnn/input_from_feature_columns/input_layer/feature_4/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/feature_4/ShapeHdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stackJdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/feature_4/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 

Ddnn/input_from_feature_columns/input_layer/feature_4/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Bdnn/input_from_feature_columns/input_layer/feature_4/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/feature_4/strided_sliceDdnn/input_from_feature_columns/input_layer/feature_4/Reshape/shape/1*
N*
_output_shapes
:*
T0
ћ
<dnn/input_from_feature_columns/input_layer/feature_4/ReshapeReshape<dnn/input_from_feature_columns/input_layer/feature_4/truedivBdnn/input_from_feature_columns/input_layer/feature_4/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ

:dnn/input_from_feature_columns/input_layer/feature_5/sub/yConst*
valueB
 *ДvA*
dtype0*
_output_shapes
: 
Ы
8dnn/input_from_feature_columns/input_layer/feature_5/subSubParseExample/ParseExample:27:dnn/input_from_feature_columns/input_layer/feature_5/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

>dnn/input_from_feature_columns/input_layer/feature_5/truediv/yConst*
valueB
 *Ыя2A*
dtype0*
_output_shapes
: 
ѓ
<dnn/input_from_feature_columns/input_layer/feature_5/truedivRealDiv8dnn/input_from_feature_columns/input_layer/feature_5/sub>dnn/input_from_feature_columns/input_layer/feature_5/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
І
:dnn/input_from_feature_columns/input_layer/feature_5/ShapeShape<dnn/input_from_feature_columns/input_layer/feature_5/truediv*
T0*
_output_shapes
:

Hdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Jdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Jdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ж
Bdnn/input_from_feature_columns/input_layer/feature_5/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/feature_5/ShapeHdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stackJdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/feature_5/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 

Ddnn/input_from_feature_columns/input_layer/feature_5/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :

Bdnn/input_from_feature_columns/input_layer/feature_5/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/feature_5/strided_sliceDdnn/input_from_feature_columns/input_layer/feature_5/Reshape/shape/1*
N*
_output_shapes
:*
T0
ћ
<dnn/input_from_feature_columns/input_layer/feature_5/ReshapeReshape<dnn/input_from_feature_columns/input_layer/feature_5/truedivBdnn/input_from_feature_columns/input_layer/feature_5/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ

:dnn/input_from_feature_columns/input_layer/feature_6/sub/yConst*
valueB
 *jјC*
dtype0*
_output_shapes
: 
Ы
8dnn/input_from_feature_columns/input_layer/feature_6/subSubParseExample/ParseExample:28:dnn/input_from_feature_columns/input_layer/feature_6/sub/y*'
_output_shapes
:џџџџџџџџџ*
T0

>dnn/input_from_feature_columns/input_layer/feature_6/truediv/yConst*
valueB
 *м)D*
dtype0*
_output_shapes
: 
ѓ
<dnn/input_from_feature_columns/input_layer/feature_6/truedivRealDiv8dnn/input_from_feature_columns/input_layer/feature_6/sub>dnn/input_from_feature_columns/input_layer/feature_6/truediv/y*'
_output_shapes
:џџџџџџџџџ*
T0
І
:dnn/input_from_feature_columns/input_layer/feature_6/ShapeShape<dnn/input_from_feature_columns/input_layer/feature_6/truediv*
T0*
_output_shapes
:

Hdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Jdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Jdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ж
Bdnn/input_from_feature_columns/input_layer/feature_6/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/feature_6/ShapeHdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stackJdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/feature_6/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0

Ddnn/input_from_feature_columns/input_layer/feature_6/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Bdnn/input_from_feature_columns/input_layer/feature_6/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/feature_6/strided_sliceDdnn/input_from_feature_columns/input_layer/feature_6/Reshape/shape/1*
T0*
N*
_output_shapes
:
ћ
<dnn/input_from_feature_columns/input_layer/feature_6/ReshapeReshape<dnn/input_from_feature_columns/input_layer/feature_6/truedivBdnn/input_from_feature_columns/input_layer/feature_6/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0
Д
Ednn/input_from_feature_columns/input_layer/feature_7_embedding/lookupStringToHashBucketFastParseExample/ParseExample:9*#
_output_shapes
:џџџџџџџџџ*
num_buckets
А
fdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
Џ
ednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0

`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SliceSliceParseExample/ParseExample:15fdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice/beginednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
Њ
`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ь
_dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/ProdProd`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Const*
T0	*
_output_shapes
: 
­
kdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Њ
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ј
cdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:15kdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2/indiceshdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
н
adnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Cast/xPack_dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Ц
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:3ParseExample/ParseExample:15adnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ђ
qdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshape/IdentityIdentityEdnn/input_from_feature_columns/input_layer/feature_7_embedding/lookup*
T0	*#
_output_shapes
:џџџџџџџџџ
Ћ
idnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 

gdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GreaterEqualGreaterEqualqdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshape/Identityidnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:џџџџџџџџџ
ћ
`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/WhereWheregdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
Л
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ч
bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/ReshapeReshape`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Wherehdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape/shape*#
_output_shapes
:џџџџџџџџџ*
T0	
Ќ
jdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

ednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_1GatherV2hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshapebdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_1/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	
Ќ
jdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 

ednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_2GatherV2qdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshape/Identitybdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_2/axis*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	

cdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/IdentityIdentityjdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
Ж
tdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Ђ
dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_1ednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/GatherV2_2cdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Identitytdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
и
dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
к
dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
к
dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Њ
dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stackdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ*
T0	*
Index0
П
wdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/CastCastdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
Ч
ydnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/UniqueUniquednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0	
З
dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Е
dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2\dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/readydnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/Uniquednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0	*
Tparams0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*'
_output_shapes
:џџџџџџџџџ*
Taxis0
б
dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitydnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:џџџџџџџџџ*
T0
О
rdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparseSparseSegmentMeandnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity{dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/Unique:1wdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
Л
jdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

ddnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_1Reshapednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2jdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ

`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/ShapeShaperdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
И
ndnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
К
pdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
К
pdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
є
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Shapendnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stackpdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_1pdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
Є
bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
ф
`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/stackPackbdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/stack/0hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ъ
_dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/TileTileddnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_1`dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/stack*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

ednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/zeros_like	ZerosLikerdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
в
Zdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weightsSelect_dnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Tileednn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/zeros_likerdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
Л
adnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Cast_1CastParseExample/ParseExample:15*
_output_shapes
:*

DstT0*

SrcT0	
В
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
Б
gdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
г
bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1Sliceadnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Cast_1hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1/begingdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
ь
bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Shape_1ShapeZdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights*
T0*
_output_shapes
:
В
hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
К
gdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
д
bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2Slicebdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Shape_1hdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2/begingdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ј
fdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ы
adnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/concatConcatV2bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_1bdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Slice_2fdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
р
ddnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_2ReshapeZdnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weightsadnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/concat*'
_output_shapes
:џџџџџџџџџ*
T0
и
Ddnn/input_from_feature_columns/input_layer/feature_7_embedding/ShapeShapeddnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_2*
_output_shapes
:*
T0

Rdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Tdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0

Tdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ш
Ldnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/feature_7_embedding/ShapeRdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stackTdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Ndnn/input_from_feature_columns/input_layer/feature_7_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
 
Ldnn/input_from_feature_columns/input_layer/feature_7_embedding/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/feature_7_embedding/strided_sliceNdnn/input_from_feature_columns/input_layer/feature_7_embedding/Reshape/shape/1*
_output_shapes
:*
T0*
N
З
Fdnn/input_from_feature_columns/input_layer/feature_7_embedding/ReshapeReshapeddnn/input_from_feature_columns/input_layer/feature_7_embedding/feature_7_embedding_weights/Reshape_2Ldnn/input_from_feature_columns/input_layer/feature_7_embedding/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0
Е
Ednn/input_from_feature_columns/input_layer/feature_8_embedding/lookupStringToHashBucketFastParseExample/ParseExample:10*#
_output_shapes
:џџџџџџџџџ*
num_buckets
А
fdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
Џ
ednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SliceSliceParseExample/ParseExample:16fdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice/beginednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
Њ
`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ь
_dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/ProdProd`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Const*
T0	*
_output_shapes
: 
­
kdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Њ
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ј
cdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:16kdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2/indiceshdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2/axis*
_output_shapes
: *
Taxis0*
Tindices0*
Tparams0	
н
adnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Cast/xPack_dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
Ц
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:4ParseExample/ParseExample:16adnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ђ
qdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshape/IdentityIdentityEdnn/input_from_feature_columns/input_layer/feature_8_embedding/lookup*#
_output_shapes
:џџџџџџџџџ*
T0	
Ћ
idnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 

gdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GreaterEqualGreaterEqualqdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshape/Identityidnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:џџџџџџџџџ
ћ
`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/WhereWheregdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
Л
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ч
bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/ReshapeReshape`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Wherehdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape/shape*#
_output_shapes
:џџџџџџџџџ*
T0	
Ќ
jdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

ednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_1GatherV2hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshapebdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_1/axis*
Tindices0	*
Tparams0	*'
_output_shapes
:џџџџџџџџџ*
Taxis0
Ќ
jdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

ednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_2GatherV2qdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshape/Identitybdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_2/axis*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	

cdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/IdentityIdentityjdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
Ж
tdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Ђ
dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_1ednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/GatherV2_2cdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Identitytdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
и
dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
к
dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
к
dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Њ
dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stackdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ*
T0	*
Index0*
shrink_axis_mask
П
wdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/CastCastdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
Ч
ydnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/UniqueUniquednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
З
dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Е
dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2\dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/readydnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/Uniquednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*
Tindices0	*
Tparams0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*'
_output_shapes
:џџџџџџџџџ
б
dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitydnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:џџџџџџџџџ*
T0
О
rdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparseSparseSegmentMeandnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity{dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/Unique:1wdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
Л
jdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

ddnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_1Reshapednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2jdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ

`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/ShapeShaperdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
И
ndnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
К
pdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
К
pdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
є
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Shapendnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stackpdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_1pdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
Є
bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
ф
`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/stackPackbdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/stack/0hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ъ
_dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/TileTileddnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_1`dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/stack*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

ednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/zeros_like	ZerosLikerdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
в
Zdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weightsSelect_dnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Tileednn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/zeros_likerdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
Л
adnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Cast_1CastParseExample/ParseExample:16*

SrcT0	*
_output_shapes
:*

DstT0
В
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
Б
gdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
г
bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1Sliceadnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Cast_1hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1/begingdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
ь
bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Shape_1ShapeZdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights*
T0*
_output_shapes
:
В
hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
К
gdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
д
bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2Slicebdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Shape_1hdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2/begingdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ј
fdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ы
adnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/concatConcatV2bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_1bdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Slice_2fdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
р
ddnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_2ReshapeZdnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weightsadnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/concat*
T0*'
_output_shapes
:џџџџџџџџџ
и
Ddnn/input_from_feature_columns/input_layer/feature_8_embedding/ShapeShapeddnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_2*
_output_shapes
:*
T0

Rdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Tdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Tdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ш
Ldnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/feature_8_embedding/ShapeRdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stackTdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ndnn/input_from_feature_columns/input_layer/feature_8_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
 
Ldnn/input_from_feature_columns/input_layer/feature_8_embedding/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/feature_8_embedding/strided_sliceNdnn/input_from_feature_columns/input_layer/feature_8_embedding/Reshape/shape/1*
_output_shapes
:*
T0*
N
З
Fdnn/input_from_feature_columns/input_layer/feature_8_embedding/ReshapeReshapeddnn/input_from_feature_columns/input_layer/feature_8_embedding/feature_8_embedding_weights/Reshape_2Ldnn/input_from_feature_columns/input_layer/feature_8_embedding/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Ж
Ednn/input_from_feature_columns/input_layer/feature_9_embedding/lookupStringToHashBucketFastParseExample/ParseExample:11*#
_output_shapes
:џџџџџџџџџ*
num_bucketsЅ
А
fdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0
Џ
ednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0

`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SliceSliceParseExample/ParseExample:17fdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice/beginednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
Њ
`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ь
_dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/ProdProd`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Const*
T0	*
_output_shapes
: 
­
kdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
Њ
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ј
cdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:17kdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2/indiceshdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
н
adnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Cast/xPack_dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
Ц
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:5ParseExample/ParseExample:17adnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ђ
qdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshape/IdentityIdentityEdnn/input_from_feature_columns/input_layer/feature_9_embedding/lookup*#
_output_shapes
:џџџџџџџџџ*
T0	
Ћ
idnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	

gdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GreaterEqualGreaterEqualqdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshape/Identityidnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GreaterEqual/y*#
_output_shapes
:џџџџџџџџџ*
T0	
ћ
`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/WhereWheregdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
Л
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ч
bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/ReshapeReshape`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Wherehdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:џџџџџџџџџ
Ќ
jdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

ednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_1GatherV2hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshapebdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:џџџџџџџџџ
Ќ
jdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0

ednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_2GatherV2qdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshape/Identitybdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:џџџџџџџџџ*
Taxis0

cdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/IdentityIdentityjdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
Ж
tdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Ђ
dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_1ednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/GatherV2_2cdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Identitytdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
T0	
и
dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
к
dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
к
dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Њ
dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stackdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ*
T0	*
Index0
П
wdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/CastCastdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
Ч
ydnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/UniqueUniquednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
З
dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
Е
dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2\dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/readydnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/Uniquednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0
б
dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitydnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:џџџџџџџџџ
О
rdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparseSparseSegmentMeandnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity{dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/Unique:1wdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
Л
jdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ   

ddnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_1Reshapednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2jdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ

`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/ShapeShaperdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
И
ndnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
К
pdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
К
pdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
є
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Shapendnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stackpdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_1pdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
Є
bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
ф
`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/stackPackbdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/stack/0hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ъ
_dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/TileTileddnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_1`dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/stack*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

ednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/zeros_like	ZerosLikerdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
в
Zdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weightsSelect_dnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Tileednn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/zeros_likerdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:џџџџџџџџџ*
T0
Л
adnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Cast_1CastParseExample/ParseExample:17*
_output_shapes
:*

DstT0*

SrcT0	
В
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
Б
gdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
г
bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1Sliceadnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Cast_1hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1/begingdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
ь
bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Shape_1ShapeZdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights*
_output_shapes
:*
T0
В
hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
К
gdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
д
bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2Slicebdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Shape_1hdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2/begingdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ј
fdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ы
adnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/concatConcatV2bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_1bdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Slice_2fdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
р
ddnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_2ReshapeZdnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weightsadnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/concat*
T0*'
_output_shapes
:џџџџџџџџџ
и
Ddnn/input_from_feature_columns/input_layer/feature_9_embedding/ShapeShapeddnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_2*
T0*
_output_shapes
:

Rdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Tdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Tdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ш
Ldnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/feature_9_embedding/ShapeRdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stackTdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Ndnn/input_from_feature_columns/input_layer/feature_9_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
 
Ldnn/input_from_feature_columns/input_layer/feature_9_embedding/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/feature_9_embedding/strided_sliceNdnn/input_from_feature_columns/input_layer/feature_9_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0
З
Fdnn/input_from_feature_columns/input_layer/feature_9_embedding/ReshapeReshapeddnn/input_from_feature_columns/input_layer/feature_9_embedding/feature_9_embedding_weights/Reshape_2Ldnn/input_from_feature_columns/input_layer/feature_9_embedding/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 


1dnn/input_from_feature_columns/input_layer/concatConcatV2=dnn/input_from_feature_columns/input_layer/feature_10/Reshape=dnn/input_from_feature_columns/input_layer/feature_11/Reshape=dnn/input_from_feature_columns/input_layer/feature_12/Reshape=dnn/input_from_feature_columns/input_layer/feature_13/ReshapeGdnn/input_from_feature_columns/input_layer/feature_14_embedding/Reshape=dnn/input_from_feature_columns/input_layer/feature_15/Reshape=dnn/input_from_feature_columns/input_layer/feature_16/Reshape=dnn/input_from_feature_columns/input_layer/feature_17/ReshapeGdnn/input_from_feature_columns/input_layer/feature_18_embedding/ReshapeFdnn/input_from_feature_columns/input_layer/feature_2_embedding/Reshape<dnn/input_from_feature_columns/input_layer/feature_3/Reshape<dnn/input_from_feature_columns/input_layer/feature_4/Reshape<dnn/input_from_feature_columns/input_layer/feature_5/Reshape<dnn/input_from_feature_columns/input_layer/feature_6/ReshapeFdnn/input_from_feature_columns/input_layer/feature_7_embedding/ReshapeFdnn/input_from_feature_columns/input_layer/feature_8_embedding/ReshapeFdnn/input_from_feature_columns/input_layer/feature_9_embedding/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
N*(
_output_shapes
:џџџџџџџџџ*
T0
Х
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:
З
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *MPО*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
З
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *MP>*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
 
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed**
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
seed2№

>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
­
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	

:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	
Я
dnn/hiddenlayer_0/kernel/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape:	*0
shared_name!dnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

@dnn/hiddenlayer_0/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
и
&dnn/hiddenlayer_0/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0
Ш
3dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:	
Ў
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
Ф
dnn/hiddenlayer_0/bias/part_0VarHandleOp*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: *
shape:*.
shared_namednn/hiddenlayer_0/bias/part_0

>dnn/hiddenlayer_0/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
: 
Ч
$dnn/hiddenlayer_0/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
dtype0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
Н
1dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
:*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0

'dnn/hiddenlayer_0/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:	
w
dnn/hiddenlayer_0/kernelIdentity'dnn/hiddenlayer_0/kernel/ReadVariableOp*
T0*
_output_shapes
:	
Ё
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*'
_output_shapes
:џџџџџџџџџ*
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

dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*'
_output_shapes
:џџџџџџџџџ
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
Ч
;dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Initializer/onesConst*
_output_shapes
:*
valueB*  ?*=
_class3
1/loc:@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
dtype0
ы
*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0VarHandleOp*
shape:*;
shared_name,*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*=
_class3
1/loc:@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
dtype0*
_output_shapes
: 
Ѕ
Kdnn/hiddenlayer_0/batchnorm_0/gamma/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
_output_shapes
: 
њ
1dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/AssignAssignVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0;dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Initializer/ones*
dtype0*=
_class3
1/loc:@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0
ф
>dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Read/ReadVariableOpReadVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*=
_class3
1/loc:@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
dtype0*
_output_shapes
:
Ц
;dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/beta/part_0
ш
)dnn/hiddenlayer_0/batchnorm_0/beta/part_0VarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
dtype0*
_output_shapes
: *
shape:*:
shared_name+)dnn/hiddenlayer_0/batchnorm_0/beta/part_0
Ѓ
Jdnn/hiddenlayer_0/batchnorm_0/beta/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
_output_shapes
: 
ї
0dnn/hiddenlayer_0/batchnorm_0/beta/part_0/AssignAssignVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0;dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Initializer/zeros*
dtype0*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/beta/part_0
с
=dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
dtype0*
_output_shapes
:
Ц
;dnn/hiddenlayer_0/batchnorm_0/moving_mean/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/moving_mean*
dtype0*
_output_shapes
:
ш
)dnn/hiddenlayer_0/batchnorm_0/moving_meanVarHandleOp*:
shared_name+)dnn/hiddenlayer_0/batchnorm_0/moving_mean*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/moving_mean*
dtype0*
_output_shapes
: *
shape:
Ѓ
Jdnn/hiddenlayer_0/batchnorm_0/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean*
_output_shapes
: 
ї
0dnn/hiddenlayer_0/batchnorm_0/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean;dnn/hiddenlayer_0/batchnorm_0/moving_mean/Initializer/zeros*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/moving_mean*
dtype0
с
=dnn/hiddenlayer_0/batchnorm_0/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean*<
_class2
0.loc:@dnn/hiddenlayer_0/batchnorm_0/moving_mean*
dtype0*
_output_shapes
:
Э
>dnn/hiddenlayer_0/batchnorm_0/moving_variance/Initializer/onesConst*
valueB*  ?*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/moving_variance*
dtype0*
_output_shapes
:
є
-dnn/hiddenlayer_0/batchnorm_0/moving_varianceVarHandleOp*>
shared_name/-dnn/hiddenlayer_0/batchnorm_0/moving_variance*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/moving_variance*
dtype0*
_output_shapes
: *
shape:
Ћ
Ndnn/hiddenlayer_0/batchnorm_0/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance*
_output_shapes
: 

4dnn/hiddenlayer_0/batchnorm_0/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance>dnn/hiddenlayer_0/batchnorm_0/moving_variance/Initializer/ones*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/moving_variance*
dtype0
э
Adnn/hiddenlayer_0/batchnorm_0/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/moving_variance*
dtype0*
_output_shapes
:

1dnn/hiddenlayer_0/batchnorm_0/beta/ReadVariableOpReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
dtype0*
_output_shapes
:

"dnn/hiddenlayer_0/batchnorm_0/betaIdentity1dnn/hiddenlayer_0/batchnorm_0/beta/ReadVariableOp*
T0*
_output_shapes
:
 
6dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variance*
dtype0*
_output_shapes
:
r
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
О
+dnn/hiddenlayer_0/batchnorm_0/batchnorm/addAdd6dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOp-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add/y*
T0*
_output_shapes
:

-dnn/hiddenlayer_0/batchnorm_0/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_0/batchnorm_0/batchnorm/add*
_output_shapes
:*
T0

2dnn/hiddenlayer_0/batchnorm_0/gamma/ReadVariableOpReadVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
dtype0*
_output_shapes
:

#dnn/hiddenlayer_0/batchnorm_0/gammaIdentity2dnn/hiddenlayer_0/batchnorm_0/gamma/ReadVariableOp*
_output_shapes
:*
T0
Ћ
+dnn/hiddenlayer_0/batchnorm_0/batchnorm/mulMul-dnn/hiddenlayer_0/batchnorm_0/batchnorm/Rsqrt#dnn/hiddenlayer_0/batchnorm_0/gamma*
T0*
_output_shapes
:
Ћ
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_1Muldnn/hiddenlayer_0/Relu+dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ

8dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_mean*
dtype0*
_output_shapes
:
Р
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_2Mul8dnn/hiddenlayer_0/batchnorm_0/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul*
T0*
_output_shapes
:
Њ
+dnn/hiddenlayer_0/batchnorm_0/batchnorm/subSub"dnn/hiddenlayer_0/batchnorm_0/beta-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_2*
T0*
_output_shapes
:
Т
-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1Add-dnn/hiddenlayer_0/batchnorm_0/batchnorm/mul_1+dnn/hiddenlayer_0/batchnorm_0/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ
~
dnn/zero_fraction/SizeSize-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1*
T0*
out_type0	*
_output_shapes
: 
c
dnn/zero_fraction/LessEqual/yConst*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 

dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 

dnn/zero_fraction/cond/SwitchSwitchdnn/zero_fraction/LessEqualdnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
m
dnn/zero_fraction/cond/switch_tIdentitydnn/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

k
dnn/zero_fraction/cond/switch_fIdentitydnn/zero_fraction/cond/Switch*
_output_shapes
: *
T0

h
dnn/zero_fraction/cond/pred_idIdentitydnn/zero_fraction/LessEqual*
_output_shapes
: *
T0


*dnn/zero_fraction/cond/count_nonzero/zerosConst ^dnn/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Я
-dnn/zero_fraction/cond/count_nonzero/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1*dnn/zero_fraction/cond/count_nonzero/zeros*
T0*'
_output_shapes
:џџџџџџџџџ

4dnn/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitch-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1dnn/zero_fraction/cond/pred_id*
T0*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
Ё
)dnn/zero_fraction/cond/count_nonzero/CastCast-dnn/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:џџџџџџџџџ*

DstT0

*dnn/zero_fraction/cond/count_nonzero/ConstConst ^dnn/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Б
2dnn/zero_fraction/cond/count_nonzero/nonzero_countSum)dnn/zero_fraction/cond/count_nonzero/Cast*dnn/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 

dnn/zero_fraction/cond/CastCast2dnn/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0

,dnn/zero_fraction/cond/count_nonzero_1/zerosConst ^dnn/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
г
/dnn/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch,dnn/zero_fraction/cond/count_nonzero_1/zeros*'
_output_shapes
:џџџџџџџџџ*
T0

6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1dnn/zero_fraction/cond/pred_id*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
T0*@
_class6
42loc:@dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1
Ѕ
+dnn/zero_fraction/cond/count_nonzero_1/CastCast/dnn/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:џџџџџџџџџ*

DstT0	

,dnn/zero_fraction/cond/count_nonzero_1/ConstConst ^dnn/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"       
З
4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countSum+dnn/zero_fraction/cond/count_nonzero_1/Cast,dnn/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Є
dnn/zero_fraction/cond/MergeMerge4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countdnn/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 

)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
{
+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
А
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
_output_shapes
: *>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0
Џ
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 
Ё
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tag-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1*
_output_shapes
: 
Х
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0
З
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *зГнО*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
З
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *зГн>*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 

Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
seed2Ч*
dtype0*
_output_shapes

:*

seed**
T0

>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
Ќ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:

:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
Ю
dnn/hiddenlayer_1/kernel/part_0VarHandleOp*0
shared_name!dnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: *
shape
:

@dnn/hiddenlayer_1/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
и
&dnn/hiddenlayer_1/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
Ч
3dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:
Ў
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
Ф
dnn/hiddenlayer_1/bias/part_0VarHandleOp*.
shared_namednn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: *
shape:

>dnn/hiddenlayer_1/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
Ч
$dnn/hiddenlayer_1/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0
Н
1dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:

'dnn/hiddenlayer_1/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:
v
dnn/hiddenlayer_1/kernelIdentity'dnn/hiddenlayer_1/kernel/ReadVariableOp*
T0*
_output_shapes

:

dnn/hiddenlayer_1/MatMulMatMul-dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1dnn/hiddenlayer_1/kernel*'
_output_shapes
:џџџџџџџџџ*
T0
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

dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*'
_output_shapes
:џџџџџџџџџ*
T0
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
Ч
;dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Initializer/onesConst*
valueB*  ?*=
_class3
1/loc:@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
dtype0*
_output_shapes
:
ы
*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0VarHandleOp*=
_class3
1/loc:@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
dtype0*
_output_shapes
: *
shape:*;
shared_name,*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0
Ѕ
Kdnn/hiddenlayer_1/batchnorm_1/gamma/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
_output_shapes
: 
њ
1dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/AssignAssignVariableOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0;dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Initializer/ones*
dtype0*=
_class3
1/loc:@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0
ф
>dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Read/ReadVariableOpReadVariableOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
dtype0*
_output_shapes
:*=
_class3
1/loc:@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0
Ц
;dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0*
_output_shapes
:
ш
)dnn/hiddenlayer_1/batchnorm_1/beta/part_0VarHandleOp*:
shared_name+)dnn/hiddenlayer_1/batchnorm_1/beta/part_0*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0*
_output_shapes
: *
shape:
Ѓ
Jdnn/hiddenlayer_1/batchnorm_1/beta/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
_output_shapes
: 
ї
0dnn/hiddenlayer_1/batchnorm_1/beta/part_0/AssignAssignVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0;dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Initializer/zeros*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0
с
=dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
_output_shapes
:*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0
Ц
;dnn/hiddenlayer_1/batchnorm_1/moving_mean/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/moving_mean*
dtype0*
_output_shapes
:
ш
)dnn/hiddenlayer_1/batchnorm_1/moving_meanVarHandleOp*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/moving_mean*
dtype0*
_output_shapes
: *
shape:*:
shared_name+)dnn/hiddenlayer_1/batchnorm_1/moving_mean
Ѓ
Jdnn/hiddenlayer_1/batchnorm_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean*
_output_shapes
: 
ї
0dnn/hiddenlayer_1/batchnorm_1/moving_mean/AssignAssignVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean;dnn/hiddenlayer_1/batchnorm_1/moving_mean/Initializer/zeros*
dtype0*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/moving_mean
с
=dnn/hiddenlayer_1/batchnorm_1/moving_mean/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean*<
_class2
0.loc:@dnn/hiddenlayer_1/batchnorm_1/moving_mean*
dtype0*
_output_shapes
:
Э
>dnn/hiddenlayer_1/batchnorm_1/moving_variance/Initializer/onesConst*
valueB*  ?*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/moving_variance*
dtype0*
_output_shapes
:
є
-dnn/hiddenlayer_1/batchnorm_1/moving_varianceVarHandleOp*>
shared_name/-dnn/hiddenlayer_1/batchnorm_1/moving_variance*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/moving_variance*
dtype0*
_output_shapes
: *
shape:
Ћ
Ndnn/hiddenlayer_1/batchnorm_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance*
_output_shapes
: 

4dnn/hiddenlayer_1/batchnorm_1/moving_variance/AssignAssignVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance>dnn/hiddenlayer_1/batchnorm_1/moving_variance/Initializer/ones*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/moving_variance*
dtype0
э
Adnn/hiddenlayer_1/batchnorm_1/moving_variance/Read/ReadVariableOpReadVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance*
_output_shapes
:*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/moving_variance*
dtype0

1dnn/hiddenlayer_1/batchnorm_1/beta/ReadVariableOpReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0*
_output_shapes
:

"dnn/hiddenlayer_1/batchnorm_1/betaIdentity1dnn/hiddenlayer_1/batchnorm_1/beta/ReadVariableOp*
T0*
_output_shapes
:
 
6dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOpReadVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variance*
dtype0*
_output_shapes
:
r
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
О
+dnn/hiddenlayer_1/batchnorm_1/batchnorm/addAdd6dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOp-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add/y*
T0*
_output_shapes
:

-dnn/hiddenlayer_1/batchnorm_1/batchnorm/RsqrtRsqrt+dnn/hiddenlayer_1/batchnorm_1/batchnorm/add*
_output_shapes
:*
T0

2dnn/hiddenlayer_1/batchnorm_1/gamma/ReadVariableOpReadVariableOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0*
dtype0*
_output_shapes
:

#dnn/hiddenlayer_1/batchnorm_1/gammaIdentity2dnn/hiddenlayer_1/batchnorm_1/gamma/ReadVariableOp*
T0*
_output_shapes
:
Ћ
+dnn/hiddenlayer_1/batchnorm_1/batchnorm/mulMul-dnn/hiddenlayer_1/batchnorm_1/batchnorm/Rsqrt#dnn/hiddenlayer_1/batchnorm_1/gamma*
T0*
_output_shapes
:
Ћ
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_1Muldnn/hiddenlayer_1/Relu+dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ*
T0

8dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOp_1ReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_mean*
dtype0*
_output_shapes
:
Р
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_2Mul8dnn/hiddenlayer_1/batchnorm_1/batchnorm/ReadVariableOp_1+dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul*
T0*
_output_shapes
:
Њ
+dnn/hiddenlayer_1/batchnorm_1/batchnorm/subSub"dnn/hiddenlayer_1/batchnorm_1/beta-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_2*
T0*
_output_shapes
:
Т
-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1Add-dnn/hiddenlayer_1/batchnorm_1/batchnorm/mul_1+dnn/hiddenlayer_1/batchnorm_1/batchnorm/sub*'
_output_shapes
:џџџџџџџџџ*
T0

dnn/zero_fraction_1/SizeSize-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1*
out_type0	*
_output_shapes
: *
T0
e
dnn/zero_fraction_1/LessEqual/yConst*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 

dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 

dnn/zero_fraction_1/cond/SwitchSwitchdnn/zero_fraction_1/LessEqualdnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_1/cond/switch_tIdentity!dnn/zero_fraction_1/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_1/cond/switch_fIdentitydnn/zero_fraction_1/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_1/cond/pred_idIdentitydnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: 

,dnn/zero_fraction_1/cond/count_nonzero/zerosConst"^dnn/zero_fraction_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
е
/dnn/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_1/cond/count_nonzero/zeros*
T0*'
_output_shapes
:џџџџџџџџџ

6dnn/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitch-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1 dnn/zero_fraction_1/cond/pred_id*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
T0*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1
Ѕ
+dnn/zero_fraction_1/cond/count_nonzero/CastCast/dnn/zero_fraction_1/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:џџџџџџџџџ*

DstT0
Ё
,dnn/zero_fraction_1/cond/count_nonzero/ConstConst"^dnn/zero_fraction_1/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
З
4dnn/zero_fraction_1/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_1/cond/count_nonzero/Cast,dnn/zero_fraction_1/cond/count_nonzero/Const*
_output_shapes
: *
T0

dnn/zero_fraction_1/cond/CastCast4dnn/zero_fraction_1/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0

.dnn/zero_fraction_1/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_1/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
й
1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_1/cond/count_nonzero_1/zeros*'
_output_shapes
:џџџџџџџџџ*
T0

8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitch-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1 dnn/zero_fraction_1/cond/pred_id*
T0*@
_class6
42loc:@dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
Љ
-dnn/zero_fraction_1/cond/count_nonzero_1/CastCast1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual*'
_output_shapes
:џџџџџџџџџ*

DstT0	*

SrcT0

Ѓ
.dnn/zero_fraction_1/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_1/cond/switch_f*
_output_shapes
:*
valueB"       *
dtype0
Н
6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_1/cond/count_nonzero_1/Cast.dnn/zero_fraction_1/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
Њ
dnn/zero_fraction_1/cond/MergeMerge6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_1/cond/Cast*
T0	*
N*
_output_shapes
: : 

*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Sizednn/zero_fraction_1/cond/Merge*
_output_shapes
: *
T0	

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
Ж
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
_output_shapes
: *
T0
 
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*
_output_shapes
: *>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0
Б
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
_output_shapes
: *
T0

$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
Ё
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tag-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1*
_output_shapes
: 
З
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
:
Љ
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *0П*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
Љ
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *0?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 

Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2*
dtype0*
_output_shapes

:*

seed*
ў
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 

7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:

3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
Й
dnn/logits/kernel/part_0VarHandleOp*)
shared_namednn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: *
shape
:

9dnn/logits/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel/part_0*
_output_shapes
: 
М
dnn/logits/kernel/part_0/AssignAssignVariableOpdnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
В
,dnn/logits/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:
 
(dnn/logits/bias/part_0/Initializer/zerosConst*
valueB*    *)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
Џ
dnn/logits/bias/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape:*'
shared_namednn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0
}
7dnn/logits/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias/part_0*
_output_shapes
: 
Ћ
dnn/logits/bias/part_0/AssignAssignVariableOpdnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*)
_class
loc:@dnn/logits/bias/part_0*
dtype0
Ј
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

dnn/logits/MatMulMatMul-dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1dnn/logits/kernel*
T0*'
_output_shapes
:џџџџџџџџџ
q
dnn/logits/bias/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
`
dnn/logits/biasIdentitydnn/logits/bias/ReadVariableOp*
T0*
_output_shapes
:
s
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*'
_output_shapes
:џџџџџџџџџ*
T0
e
dnn/zero_fraction_2/SizeSizednn/logits/BiasAdd*
_output_shapes
: *
T0*
out_type0	
e
dnn/zero_fraction_2/LessEqual/yConst*
_output_shapes
: *
valueB	 Rџџџџ*
dtype0	

dnn/zero_fraction_2/LessEqual	LessEqualdnn/zero_fraction_2/Sizednn/zero_fraction_2/LessEqual/y*
T0	*
_output_shapes
: 

dnn/zero_fraction_2/cond/SwitchSwitchdnn/zero_fraction_2/LessEqualdnn/zero_fraction_2/LessEqual*
T0
*
_output_shapes
: : 
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

,dnn/zero_fraction_2/cond/count_nonzero/zerosConst"^dnn/zero_fraction_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
е
/dnn/zero_fraction_2/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_2/cond/count_nonzero/zeros*'
_output_shapes
:џџџџџџџџџ*
T0
т
6dnn/zero_fraction_2/cond/count_nonzero/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_2/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
Ѕ
+dnn/zero_fraction_2/cond/count_nonzero/CastCast/dnn/zero_fraction_2/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:џџџџџџџџџ*

DstT0
Ё
,dnn/zero_fraction_2/cond/count_nonzero/ConstConst"^dnn/zero_fraction_2/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
З
4dnn/zero_fraction_2/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_2/cond/count_nonzero/Cast,dnn/zero_fraction_2/cond/count_nonzero/Const*
_output_shapes
: *
T0

dnn/zero_fraction_2/cond/CastCast4dnn/zero_fraction_2/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	

.dnn/zero_fraction_2/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_2/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
й
1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_2/cond/count_nonzero_1/zeros*'
_output_shapes
:џџџџџџџџџ*
T0
ф
8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_2/cond/pred_id*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
T0*%
_class
loc:@dnn/logits/BiasAdd
Љ
-dnn/zero_fraction_2/cond/count_nonzero_1/CastCast1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:џџџџџџџџџ*

DstT0	
Ѓ
.dnn/zero_fraction_2/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_2/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Н
6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_2/cond/count_nonzero_1/Cast.dnn/zero_fraction_2/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Њ
dnn/zero_fraction_2/cond/MergeMerge6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_2/cond/Cast*
T0	*
N*
_output_shapes
: : 

*dnn/zero_fraction_2/counts_to_fraction/subSubdnn/zero_fraction_2/Sizednn/zero_fraction_2/cond/Merge*
_output_shapes
: *
T0	

+dnn/zero_fraction_2/counts_to_fraction/CastCast*dnn/zero_fraction_2/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	

-dnn/zero_fraction_2/counts_to_fraction/Cast_1Castdnn/zero_fraction_2/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ж
.dnn/zero_fraction_2/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_2/counts_to_fraction/Cast-dnn/zero_fraction_2/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_2/fractionIdentity.dnn/zero_fraction_2/counts_to_fraction/truediv*
_output_shapes
: *
T0

+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
Ѓ
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/fraction*
T0*
_output_shapes
: 
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
e
 ExponentialDecay_1/learning_rateConst*
valueB
 *џцл.*
dtype0*
_output_shapes
: 
\
ExponentialDecay_1/Cast/xConst*
value
B :'*
dtype0*
_output_shapes
: 
j
ExponentialDecay_1/CastCastExponentialDecay_1/Cast/x*

SrcT0*
_output_shapes
: *

DstT0
`
ExponentialDecay_1/Cast_1/xConst*
valueB
 *Тu?*
dtype0*
_output_shapes
: 
c
ExponentialDecay_1/Cast_2Castglobal_step/read*

SrcT0	*
_output_shapes
: *

DstT0
z
ExponentialDecay_1/truedivRealDivExponentialDecay_1/Cast_2ExponentialDecay_1/Cast*
T0*
_output_shapes
: 
w
ExponentialDecay_1/PowPowExponentialDecay_1/Cast_1/xExponentialDecay_1/truediv*
T0*
_output_shapes
: 
t
ExponentialDecay_1Mul ExponentialDecay_1/learning_rateExponentialDecay_1/Pow*
_output_shapes
: *
T0
ж
?linear/linear_model/feature_10/weights/part_0/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@linear/linear_model/feature_10/weights/part_0*
dtype0*
_output_shapes

:
ј
-linear/linear_model/feature_10/weights/part_0VarHandleOp*@
_class6
42loc:@linear/linear_model/feature_10/weights/part_0*
dtype0*
_output_shapes
: *
shape
:*>
shared_name/-linear/linear_model/feature_10/weights/part_0
Ћ
Nlinear/linear_model/feature_10/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp-linear/linear_model/feature_10/weights/part_0*
_output_shapes
: 

4linear/linear_model/feature_10/weights/part_0/AssignAssignVariableOp-linear/linear_model/feature_10/weights/part_0?linear/linear_model/feature_10/weights/part_0/Initializer/zeros*@
_class6
42loc:@linear/linear_model/feature_10/weights/part_0*
dtype0
ё
Alinear/linear_model/feature_10/weights/part_0/Read/ReadVariableOpReadVariableOp-linear/linear_model/feature_10/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_10/weights/part_0*
dtype0*
_output_shapes

:
ж
?linear/linear_model/feature_11/weights/part_0/Initializer/zerosConst*
_output_shapes

:*
valueB*    *@
_class6
42loc:@linear/linear_model/feature_11/weights/part_0*
dtype0
ј
-linear/linear_model/feature_11/weights/part_0VarHandleOp*>
shared_name/-linear/linear_model/feature_11/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_11/weights/part_0*
dtype0*
_output_shapes
: *
shape
:
Ћ
Nlinear/linear_model/feature_11/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp-linear/linear_model/feature_11/weights/part_0*
_output_shapes
: 

4linear/linear_model/feature_11/weights/part_0/AssignAssignVariableOp-linear/linear_model/feature_11/weights/part_0?linear/linear_model/feature_11/weights/part_0/Initializer/zeros*@
_class6
42loc:@linear/linear_model/feature_11/weights/part_0*
dtype0
ё
Alinear/linear_model/feature_11/weights/part_0/Read/ReadVariableOpReadVariableOp-linear/linear_model/feature_11/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_11/weights/part_0*
dtype0*
_output_shapes

:
ж
?linear/linear_model/feature_12/weights/part_0/Initializer/zerosConst*
_output_shapes

:*
valueB*    *@
_class6
42loc:@linear/linear_model/feature_12/weights/part_0*
dtype0
ј
-linear/linear_model/feature_12/weights/part_0VarHandleOp*>
shared_name/-linear/linear_model/feature_12/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_12/weights/part_0*
dtype0*
_output_shapes
: *
shape
:
Ћ
Nlinear/linear_model/feature_12/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp-linear/linear_model/feature_12/weights/part_0*
_output_shapes
: 

4linear/linear_model/feature_12/weights/part_0/AssignAssignVariableOp-linear/linear_model/feature_12/weights/part_0?linear/linear_model/feature_12/weights/part_0/Initializer/zeros*@
_class6
42loc:@linear/linear_model/feature_12/weights/part_0*
dtype0
ё
Alinear/linear_model/feature_12/weights/part_0/Read/ReadVariableOpReadVariableOp-linear/linear_model/feature_12/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_12/weights/part_0*
dtype0*
_output_shapes

:
ж
?linear/linear_model/feature_13/weights/part_0/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@linear/linear_model/feature_13/weights/part_0*
dtype0*
_output_shapes

:
ј
-linear/linear_model/feature_13/weights/part_0VarHandleOp*
shape
:*>
shared_name/-linear/linear_model/feature_13/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_13/weights/part_0*
dtype0*
_output_shapes
: 
Ћ
Nlinear/linear_model/feature_13/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp-linear/linear_model/feature_13/weights/part_0*
_output_shapes
: 

4linear/linear_model/feature_13/weights/part_0/AssignAssignVariableOp-linear/linear_model/feature_13/weights/part_0?linear/linear_model/feature_13/weights/part_0/Initializer/zeros*@
_class6
42loc:@linear/linear_model/feature_13/weights/part_0*
dtype0
ё
Alinear/linear_model/feature_13/weights/part_0/Read/ReadVariableOpReadVariableOp-linear/linear_model/feature_13/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_13/weights/part_0*
dtype0*
_output_shapes

:

dlinear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"m      *T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
ў
clinear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

elinear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *ЭЬL>*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

nlinear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaldlinear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*
seed2ю*
dtype0*
_output_shapes

:m*

seed**
T0
П
blinear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulnlinear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalelinear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*
_output_shapes

:m
­
^linear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normalAddblinear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulclinear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*
_output_shapes

:m
ч
Alinear/linear_model/feature_14_embedding/embedding_weights/part_0
VariableV2*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:m*
shape
:m
є
Hlinear/linear_model/feature_14_embedding/embedding_weights/part_0/AssignAssignAlinear/linear_model/feature_14_embedding/embedding_weights/part_0^linear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*
_output_shapes

:m

Flinear/linear_model/feature_14_embedding/embedding_weights/part_0/readIdentityAlinear/linear_model/feature_14_embedding/embedding_weights/part_0*
T0*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*
_output_shapes

:m
ъ
Ilinear/linear_model/feature_14_embedding/weights/part_0/Initializer/zerosConst*
valueB*    *J
_class@
><loc:@linear/linear_model/feature_14_embedding/weights/part_0*
dtype0*
_output_shapes

:

7linear/linear_model/feature_14_embedding/weights/part_0VarHandleOp*H
shared_name97linear/linear_model/feature_14_embedding/weights/part_0*J
_class@
><loc:@linear/linear_model/feature_14_embedding/weights/part_0*
dtype0*
_output_shapes
: *
shape
:
П
Xlinear/linear_model/feature_14_embedding/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp7linear/linear_model/feature_14_embedding/weights/part_0*
_output_shapes
: 
Џ
>linear/linear_model/feature_14_embedding/weights/part_0/AssignAssignVariableOp7linear/linear_model/feature_14_embedding/weights/part_0Ilinear/linear_model/feature_14_embedding/weights/part_0/Initializer/zeros*J
_class@
><loc:@linear/linear_model/feature_14_embedding/weights/part_0*
dtype0

Klinear/linear_model/feature_14_embedding/weights/part_0/Read/ReadVariableOpReadVariableOp7linear/linear_model/feature_14_embedding/weights/part_0*J
_class@
><loc:@linear/linear_model/feature_14_embedding/weights/part_0*
dtype0*
_output_shapes

:
ж
?linear/linear_model/feature_15/weights/part_0/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@linear/linear_model/feature_15/weights/part_0*
dtype0*
_output_shapes

:
ј
-linear/linear_model/feature_15/weights/part_0VarHandleOp*
shape
:*>
shared_name/-linear/linear_model/feature_15/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_15/weights/part_0*
dtype0*
_output_shapes
: 
Ћ
Nlinear/linear_model/feature_15/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp-linear/linear_model/feature_15/weights/part_0*
_output_shapes
: 

4linear/linear_model/feature_15/weights/part_0/AssignAssignVariableOp-linear/linear_model/feature_15/weights/part_0?linear/linear_model/feature_15/weights/part_0/Initializer/zeros*
dtype0*@
_class6
42loc:@linear/linear_model/feature_15/weights/part_0
ё
Alinear/linear_model/feature_15/weights/part_0/Read/ReadVariableOpReadVariableOp-linear/linear_model/feature_15/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_15/weights/part_0*
dtype0*
_output_shapes

:
ж
?linear/linear_model/feature_16/weights/part_0/Initializer/zerosConst*
_output_shapes

:*
valueB*    *@
_class6
42loc:@linear/linear_model/feature_16/weights/part_0*
dtype0
ј
-linear/linear_model/feature_16/weights/part_0VarHandleOp*>
shared_name/-linear/linear_model/feature_16/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_16/weights/part_0*
dtype0*
_output_shapes
: *
shape
:
Ћ
Nlinear/linear_model/feature_16/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp-linear/linear_model/feature_16/weights/part_0*
_output_shapes
: 

4linear/linear_model/feature_16/weights/part_0/AssignAssignVariableOp-linear/linear_model/feature_16/weights/part_0?linear/linear_model/feature_16/weights/part_0/Initializer/zeros*@
_class6
42loc:@linear/linear_model/feature_16/weights/part_0*
dtype0
ё
Alinear/linear_model/feature_16/weights/part_0/Read/ReadVariableOpReadVariableOp-linear/linear_model/feature_16/weights/part_0*
_output_shapes

:*@
_class6
42loc:@linear/linear_model/feature_16/weights/part_0*
dtype0
ж
?linear/linear_model/feature_17/weights/part_0/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *@
_class6
42loc:@linear/linear_model/feature_17/weights/part_0
ј
-linear/linear_model/feature_17/weights/part_0VarHandleOp*>
shared_name/-linear/linear_model/feature_17/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_17/weights/part_0*
dtype0*
_output_shapes
: *
shape
:
Ћ
Nlinear/linear_model/feature_17/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp-linear/linear_model/feature_17/weights/part_0*
_output_shapes
: 

4linear/linear_model/feature_17/weights/part_0/AssignAssignVariableOp-linear/linear_model/feature_17/weights/part_0?linear/linear_model/feature_17/weights/part_0/Initializer/zeros*@
_class6
42loc:@linear/linear_model/feature_17/weights/part_0*
dtype0
ё
Alinear/linear_model/feature_17/weights/part_0/Read/ReadVariableOpReadVariableOp-linear/linear_model/feature_17/weights/part_0*@
_class6
42loc:@linear/linear_model/feature_17/weights/part_0*
dtype0*
_output_shapes

:

dlinear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"       *T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
ў
clinear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

elinear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *ѕыj>*T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

nlinear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaldlinear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
_output_shapes

: *

seed**
T0*T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*
seed2*
dtype0
П
blinear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulnlinear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalelinear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*
_output_shapes

: 
­
^linear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normalAddblinear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulclinear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*
_output_shapes

: 
ч
Alinear/linear_model/feature_18_embedding/embedding_weights/part_0
VariableV2*
_output_shapes

: *
shape
: *T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*
dtype0
є
Hlinear/linear_model/feature_18_embedding/embedding_weights/part_0/AssignAssignAlinear/linear_model/feature_18_embedding/embedding_weights/part_0^linear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*
_output_shapes

: 

Flinear/linear_model/feature_18_embedding/embedding_weights/part_0/readIdentityAlinear/linear_model/feature_18_embedding/embedding_weights/part_0*
T0*T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*
_output_shapes

: 
ъ
Ilinear/linear_model/feature_18_embedding/weights/part_0/Initializer/zerosConst*
valueB*    *J
_class@
><loc:@linear/linear_model/feature_18_embedding/weights/part_0*
dtype0*
_output_shapes

:

7linear/linear_model/feature_18_embedding/weights/part_0VarHandleOp*J
_class@
><loc:@linear/linear_model/feature_18_embedding/weights/part_0*
dtype0*
_output_shapes
: *
shape
:*H
shared_name97linear/linear_model/feature_18_embedding/weights/part_0
П
Xlinear/linear_model/feature_18_embedding/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp7linear/linear_model/feature_18_embedding/weights/part_0*
_output_shapes
: 
Џ
>linear/linear_model/feature_18_embedding/weights/part_0/AssignAssignVariableOp7linear/linear_model/feature_18_embedding/weights/part_0Ilinear/linear_model/feature_18_embedding/weights/part_0/Initializer/zeros*J
_class@
><loc:@linear/linear_model/feature_18_embedding/weights/part_0*
dtype0

Klinear/linear_model/feature_18_embedding/weights/part_0/Read/ReadVariableOpReadVariableOp7linear/linear_model/feature_18_embedding/weights/part_0*J
_class@
><loc:@linear/linear_model/feature_18_embedding/weights/part_0*
dtype0*
_output_shapes

:

clinear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"      *S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
ќ
blinear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
ў
dlinear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *ЭЬL>*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

mlinear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalclinear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:	*

seed**
T0*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
seed2
М
alinear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulmlinear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormaldlinear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	
Њ
]linear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normalAddalinear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulblinear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	*
T0
ч
@linear/linear_model/feature_2_embedding/embedding_weights/part_0
VariableV2*
shape:	*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:	
ё
Glinear/linear_model/feature_2_embedding/embedding_weights/part_0/AssignAssign@linear/linear_model/feature_2_embedding/embedding_weights/part_0]linear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	

Elinear/linear_model/feature_2_embedding/embedding_weights/part_0/readIdentity@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
T0*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	
ш
Hlinear/linear_model/feature_2_embedding/weights/part_0/Initializer/zerosConst*
valueB*    *I
_class?
=;loc:@linear/linear_model/feature_2_embedding/weights/part_0*
dtype0*
_output_shapes

:

6linear/linear_model/feature_2_embedding/weights/part_0VarHandleOp*I
_class?
=;loc:@linear/linear_model/feature_2_embedding/weights/part_0*
dtype0*
_output_shapes
: *
shape
:*G
shared_name86linear/linear_model/feature_2_embedding/weights/part_0
Н
Wlinear/linear_model/feature_2_embedding/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp6linear/linear_model/feature_2_embedding/weights/part_0*
_output_shapes
: 
Ћ
=linear/linear_model/feature_2_embedding/weights/part_0/AssignAssignVariableOp6linear/linear_model/feature_2_embedding/weights/part_0Hlinear/linear_model/feature_2_embedding/weights/part_0/Initializer/zeros*I
_class?
=;loc:@linear/linear_model/feature_2_embedding/weights/part_0*
dtype0

Jlinear/linear_model/feature_2_embedding/weights/part_0/Read/ReadVariableOpReadVariableOp6linear/linear_model/feature_2_embedding/weights/part_0*I
_class?
=;loc:@linear/linear_model/feature_2_embedding/weights/part_0*
dtype0*
_output_shapes

:
д
>linear/linear_model/feature_3/weights/part_0/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *?
_class5
31loc:@linear/linear_model/feature_3/weights/part_0
ѕ
,linear/linear_model/feature_3/weights/part_0VarHandleOp*=
shared_name.,linear/linear_model/feature_3/weights/part_0*?
_class5
31loc:@linear/linear_model/feature_3/weights/part_0*
dtype0*
_output_shapes
: *
shape
:
Љ
Mlinear/linear_model/feature_3/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/feature_3/weights/part_0*
_output_shapes
: 

3linear/linear_model/feature_3/weights/part_0/AssignAssignVariableOp,linear/linear_model/feature_3/weights/part_0>linear/linear_model/feature_3/weights/part_0/Initializer/zeros*?
_class5
31loc:@linear/linear_model/feature_3/weights/part_0*
dtype0
ю
@linear/linear_model/feature_3/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/feature_3/weights/part_0*?
_class5
31loc:@linear/linear_model/feature_3/weights/part_0*
dtype0*
_output_shapes

:
д
>linear/linear_model/feature_4/weights/part_0/Initializer/zerosConst*
valueB*    *?
_class5
31loc:@linear/linear_model/feature_4/weights/part_0*
dtype0*
_output_shapes

:
ѕ
,linear/linear_model/feature_4/weights/part_0VarHandleOp*
shape
:*=
shared_name.,linear/linear_model/feature_4/weights/part_0*?
_class5
31loc:@linear/linear_model/feature_4/weights/part_0*
dtype0*
_output_shapes
: 
Љ
Mlinear/linear_model/feature_4/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/feature_4/weights/part_0*
_output_shapes
: 

3linear/linear_model/feature_4/weights/part_0/AssignAssignVariableOp,linear/linear_model/feature_4/weights/part_0>linear/linear_model/feature_4/weights/part_0/Initializer/zeros*?
_class5
31loc:@linear/linear_model/feature_4/weights/part_0*
dtype0
ю
@linear/linear_model/feature_4/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/feature_4/weights/part_0*
_output_shapes

:*?
_class5
31loc:@linear/linear_model/feature_4/weights/part_0*
dtype0
д
>linear/linear_model/feature_5/weights/part_0/Initializer/zerosConst*
valueB*    *?
_class5
31loc:@linear/linear_model/feature_5/weights/part_0*
dtype0*
_output_shapes

:
ѕ
,linear/linear_model/feature_5/weights/part_0VarHandleOp*
_output_shapes
: *
shape
:*=
shared_name.,linear/linear_model/feature_5/weights/part_0*?
_class5
31loc:@linear/linear_model/feature_5/weights/part_0*
dtype0
Љ
Mlinear/linear_model/feature_5/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/feature_5/weights/part_0*
_output_shapes
: 

3linear/linear_model/feature_5/weights/part_0/AssignAssignVariableOp,linear/linear_model/feature_5/weights/part_0>linear/linear_model/feature_5/weights/part_0/Initializer/zeros*?
_class5
31loc:@linear/linear_model/feature_5/weights/part_0*
dtype0
ю
@linear/linear_model/feature_5/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/feature_5/weights/part_0*
dtype0*
_output_shapes

:*?
_class5
31loc:@linear/linear_model/feature_5/weights/part_0
д
>linear/linear_model/feature_6/weights/part_0/Initializer/zerosConst*
valueB*    *?
_class5
31loc:@linear/linear_model/feature_6/weights/part_0*
dtype0*
_output_shapes

:
ѕ
,linear/linear_model/feature_6/weights/part_0VarHandleOp*
shape
:*=
shared_name.,linear/linear_model/feature_6/weights/part_0*?
_class5
31loc:@linear/linear_model/feature_6/weights/part_0*
dtype0*
_output_shapes
: 
Љ
Mlinear/linear_model/feature_6/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/feature_6/weights/part_0*
_output_shapes
: 

3linear/linear_model/feature_6/weights/part_0/AssignAssignVariableOp,linear/linear_model/feature_6/weights/part_0>linear/linear_model/feature_6/weights/part_0/Initializer/zeros*?
_class5
31loc:@linear/linear_model/feature_6/weights/part_0*
dtype0
ю
@linear/linear_model/feature_6/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/feature_6/weights/part_0*?
_class5
31loc:@linear/linear_model/feature_6/weights/part_0*
dtype0*
_output_shapes

:

clinear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"      *S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
ќ
blinear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
ў
dlinear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *.љф>*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0

mlinear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalclinear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
T0*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0*
seed2Л*
dtype0*
_output_shapes

:*

seed*
Л
alinear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulmlinear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormaldlinear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0*
_output_shapes

:
Љ
]linear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normalAddalinear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulblinear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
_output_shapes

:*
T0*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0
х
@linear/linear_model/feature_7_embedding/embedding_weights/part_0
VariableV2*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:*
shape
:
№
Glinear/linear_model/feature_7_embedding/embedding_weights/part_0/AssignAssign@linear/linear_model/feature_7_embedding/embedding_weights/part_0]linear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0*
_output_shapes

:

Elinear/linear_model/feature_7_embedding/embedding_weights/part_0/readIdentity@linear/linear_model/feature_7_embedding/embedding_weights/part_0*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0*
_output_shapes

:*
T0
ш
Hlinear/linear_model/feature_7_embedding/weights/part_0/Initializer/zerosConst*
valueB*    *I
_class?
=;loc:@linear/linear_model/feature_7_embedding/weights/part_0*
dtype0*
_output_shapes

:

6linear/linear_model/feature_7_embedding/weights/part_0VarHandleOp*I
_class?
=;loc:@linear/linear_model/feature_7_embedding/weights/part_0*
dtype0*
_output_shapes
: *
shape
:*G
shared_name86linear/linear_model/feature_7_embedding/weights/part_0
Н
Wlinear/linear_model/feature_7_embedding/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp6linear/linear_model/feature_7_embedding/weights/part_0*
_output_shapes
: 
Ћ
=linear/linear_model/feature_7_embedding/weights/part_0/AssignAssignVariableOp6linear/linear_model/feature_7_embedding/weights/part_0Hlinear/linear_model/feature_7_embedding/weights/part_0/Initializer/zeros*
dtype0*I
_class?
=;loc:@linear/linear_model/feature_7_embedding/weights/part_0

Jlinear/linear_model/feature_7_embedding/weights/part_0/Read/ReadVariableOpReadVariableOp6linear/linear_model/feature_7_embedding/weights/part_0*I
_class?
=;loc:@linear/linear_model/feature_7_embedding/weights/part_0*
dtype0*
_output_shapes

:

clinear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
valueB"      *S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:
ќ
blinear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
ў
dlinear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *ѕыj>*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

mlinear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalclinear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
seed2Щ*
dtype0*
_output_shapes

:*

seed**
T0*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0
Л
alinear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulmlinear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormaldlinear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
_output_shapes

:*
T0*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0
Љ
]linear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normalAddalinear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulblinear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*
_output_shapes

:
х
@linear/linear_model/feature_8_embedding/embedding_weights/part_0
VariableV2*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:*
shape
:
№
Glinear/linear_model/feature_8_embedding/embedding_weights/part_0/AssignAssign@linear/linear_model/feature_8_embedding/embedding_weights/part_0]linear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal*
_output_shapes

:*
T0*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0

Elinear/linear_model/feature_8_embedding/embedding_weights/part_0/readIdentity@linear/linear_model/feature_8_embedding/embedding_weights/part_0*
T0*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*
_output_shapes

:
ш
Hlinear/linear_model/feature_8_embedding/weights/part_0/Initializer/zerosConst*
valueB*    *I
_class?
=;loc:@linear/linear_model/feature_8_embedding/weights/part_0*
dtype0*
_output_shapes

:

6linear/linear_model/feature_8_embedding/weights/part_0VarHandleOp*I
_class?
=;loc:@linear/linear_model/feature_8_embedding/weights/part_0*
dtype0*
_output_shapes
: *
shape
:*G
shared_name86linear/linear_model/feature_8_embedding/weights/part_0
Н
Wlinear/linear_model/feature_8_embedding/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp6linear/linear_model/feature_8_embedding/weights/part_0*
_output_shapes
: 
Ћ
=linear/linear_model/feature_8_embedding/weights/part_0/AssignAssignVariableOp6linear/linear_model/feature_8_embedding/weights/part_0Hlinear/linear_model/feature_8_embedding/weights/part_0/Initializer/zeros*I
_class?
=;loc:@linear/linear_model/feature_8_embedding/weights/part_0*
dtype0

Jlinear/linear_model/feature_8_embedding/weights/part_0/Read/ReadVariableOpReadVariableOp6linear/linear_model/feature_8_embedding/weights/part_0*I
_class?
=;loc:@linear/linear_model/feature_8_embedding/weights/part_0*
dtype0*
_output_shapes

:

clinear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"Ѕ      *S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0*
dtype0
ќ
blinear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
valueB
 *    *S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
ў
dlinear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *ЭЬL>*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0*
dtype0

mlinear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalclinear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:	Ѕ*

seed**
T0*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0*
seed2з
М
alinear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulmlinear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormaldlinear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0*
_output_shapes
:	Ѕ
Њ
]linear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normalAddalinear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulblinear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0*
_output_shapes
:	Ѕ
ч
@linear/linear_model/feature_9_embedding/embedding_weights/part_0
VariableV2*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
:	Ѕ*
shape:	Ѕ
ё
Glinear/linear_model/feature_9_embedding/embedding_weights/part_0/AssignAssign@linear/linear_model/feature_9_embedding/embedding_weights/part_0]linear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0*
_output_shapes
:	Ѕ

Elinear/linear_model/feature_9_embedding/embedding_weights/part_0/readIdentity@linear/linear_model/feature_9_embedding/embedding_weights/part_0*
_output_shapes
:	Ѕ*
T0*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0
ш
Hlinear/linear_model/feature_9_embedding/weights/part_0/Initializer/zerosConst*
valueB*    *I
_class?
=;loc:@linear/linear_model/feature_9_embedding/weights/part_0*
dtype0*
_output_shapes

:

6linear/linear_model/feature_9_embedding/weights/part_0VarHandleOp*G
shared_name86linear/linear_model/feature_9_embedding/weights/part_0*I
_class?
=;loc:@linear/linear_model/feature_9_embedding/weights/part_0*
dtype0*
_output_shapes
: *
shape
:
Н
Wlinear/linear_model/feature_9_embedding/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp6linear/linear_model/feature_9_embedding/weights/part_0*
_output_shapes
: 
Ћ
=linear/linear_model/feature_9_embedding/weights/part_0/AssignAssignVariableOp6linear/linear_model/feature_9_embedding/weights/part_0Hlinear/linear_model/feature_9_embedding/weights/part_0/Initializer/zeros*
dtype0*I
_class?
=;loc:@linear/linear_model/feature_9_embedding/weights/part_0

Jlinear/linear_model/feature_9_embedding/weights/part_0/Read/ReadVariableOpReadVariableOp6linear/linear_model/feature_9_embedding/weights/part_0*
_output_shapes

:*I
_class?
=;loc:@linear/linear_model/feature_9_embedding/weights/part_0*
dtype0
Т
9linear/linear_model/bias_weights/part_0/Initializer/zerosConst*
valueB*    *:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
т
'linear/linear_model/bias_weights/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape:*8
shared_name)'linear/linear_model/bias_weights/part_0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0

Hlinear/linear_model/bias_weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/bias_weights/part_0*
_output_shapes
: 
я
.linear/linear_model/bias_weights/part_0/AssignAssignVariableOp'linear/linear_model/bias_weights/part_09linear/linear_model/bias_weights/part_0/Initializer/zeros*
dtype0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0
л
;linear/linear_model/bias_weights/part_0/Read/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:

>linear/linear_model/linear_model/linear_model/feature_10/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *-N
г
<linear/linear_model/linear_model/linear_model/feature_10/subSubParseExample/ParseExample:18>linear/linear_model/linear_model/linear_model/feature_10/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

Blinear/linear_model/linear_model/linear_model/feature_10/truediv/yConst*
valueB
 *ЎАN*
dtype0*
_output_shapes
: 
џ
@linear/linear_model/linear_model/linear_model/feature_10/truedivRealDiv<linear/linear_model/linear_model/linear_model/feature_10/subBlinear/linear_model/linear_model/linear_model/feature_10/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
>linear/linear_model/linear_model/linear_model/feature_10/ShapeShape@linear/linear_model/linear_model/linear_model/feature_10/truediv*
T0*
_output_shapes
:

Llinear/linear_model/linear_model/linear_model/feature_10/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_10/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_10/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ъ
Flinear/linear_model/linear_model/linear_model/feature_10/strided_sliceStridedSlice>linear/linear_model/linear_model/linear_model/feature_10/ShapeLlinear/linear_model/linear_model/linear_model/feature_10/strided_slice/stackNlinear/linear_model/linear_model/linear_model/feature_10/strided_slice/stack_1Nlinear/linear_model/linear_model/linear_model/feature_10/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 

Hlinear/linear_model/linear_model/linear_model/feature_10/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :

Flinear/linear_model/linear_model/linear_model/feature_10/Reshape/shapePackFlinear/linear_model/linear_model/linear_model/feature_10/strided_sliceHlinear/linear_model/linear_model/linear_model/feature_10/Reshape/shape/1*
T0*
N*
_output_shapes
:

@linear/linear_model/linear_model/linear_model/feature_10/ReshapeReshape@linear/linear_model/linear_model/linear_model/feature_10/truedivFlinear/linear_model/linear_model/linear_model/feature_10/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
5linear/linear_model/feature_10/weights/ReadVariableOpReadVariableOp-linear/linear_model/feature_10/weights/part_0*
dtype0*
_output_shapes

:

&linear/linear_model/feature_10/weightsIdentity5linear/linear_model/feature_10/weights/ReadVariableOp*
_output_shapes

:*
T0
ы
Elinear/linear_model/linear_model/linear_model/feature_10/weighted_sumMatMul@linear/linear_model/linear_model/linear_model/feature_10/Reshape&linear/linear_model/feature_10/weights*'
_output_shapes
:џџџџџџџџџ*
T0

>linear/linear_model/linear_model/linear_model/feature_11/sub/yConst*
valueB
 *Мц?*
dtype0*
_output_shapes
: 
г
<linear/linear_model/linear_model/linear_model/feature_11/subSubParseExample/ParseExample:19>linear/linear_model/linear_model/linear_model/feature_11/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

Blinear/linear_model/linear_model/linear_model/feature_11/truediv/yConst*
valueB
 * д <*
dtype0*
_output_shapes
: 
џ
@linear/linear_model/linear_model/linear_model/feature_11/truedivRealDiv<linear/linear_model/linear_model/linear_model/feature_11/subBlinear/linear_model/linear_model/linear_model/feature_11/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
>linear/linear_model/linear_model/linear_model/feature_11/ShapeShape@linear/linear_model/linear_model/linear_model/feature_11/truediv*
T0*
_output_shapes
:

Llinear/linear_model/linear_model/linear_model/feature_11/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_11/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_11/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ъ
Flinear/linear_model/linear_model/linear_model/feature_11/strided_sliceStridedSlice>linear/linear_model/linear_model/linear_model/feature_11/ShapeLlinear/linear_model/linear_model/linear_model/feature_11/strided_slice/stackNlinear/linear_model/linear_model/linear_model/feature_11/strided_slice/stack_1Nlinear/linear_model/linear_model/linear_model/feature_11/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 

Hlinear/linear_model/linear_model/linear_model/feature_11/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Flinear/linear_model/linear_model/linear_model/feature_11/Reshape/shapePackFlinear/linear_model/linear_model/linear_model/feature_11/strided_sliceHlinear/linear_model/linear_model/linear_model/feature_11/Reshape/shape/1*
T0*
N*
_output_shapes
:

@linear/linear_model/linear_model/linear_model/feature_11/ReshapeReshape@linear/linear_model/linear_model/linear_model/feature_11/truedivFlinear/linear_model/linear_model/linear_model/feature_11/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
5linear/linear_model/feature_11/weights/ReadVariableOpReadVariableOp-linear/linear_model/feature_11/weights/part_0*
dtype0*
_output_shapes

:

&linear/linear_model/feature_11/weightsIdentity5linear/linear_model/feature_11/weights/ReadVariableOp*
T0*
_output_shapes

:
ы
Elinear/linear_model/linear_model/linear_model/feature_11/weighted_sumMatMul@linear/linear_model/linear_model/linear_model/feature_11/Reshape&linear/linear_model/feature_11/weights*
T0*'
_output_shapes
:џџџџџџџџџ

>linear/linear_model/linear_model/linear_model/feature_12/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *ђ№@
г
<linear/linear_model/linear_model/linear_model/feature_12/subSubParseExample/ParseExample:20>linear/linear_model/linear_model/linear_model/feature_12/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

Blinear/linear_model/linear_model/linear_model/feature_12/truediv/yConst*
valueB
 *Ч_B*
dtype0*
_output_shapes
: 
џ
@linear/linear_model/linear_model/linear_model/feature_12/truedivRealDiv<linear/linear_model/linear_model/linear_model/feature_12/subBlinear/linear_model/linear_model/linear_model/feature_12/truediv/y*'
_output_shapes
:џџџџџџџџџ*
T0
Ў
>linear/linear_model/linear_model/linear_model/feature_12/ShapeShape@linear/linear_model/linear_model/linear_model/feature_12/truediv*
T0*
_output_shapes
:

Llinear/linear_model/linear_model/linear_model/feature_12/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_12/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_12/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ъ
Flinear/linear_model/linear_model/linear_model/feature_12/strided_sliceStridedSlice>linear/linear_model/linear_model/linear_model/feature_12/ShapeLlinear/linear_model/linear_model/linear_model/feature_12/strided_slice/stackNlinear/linear_model/linear_model/linear_model/feature_12/strided_slice/stack_1Nlinear/linear_model/linear_model/linear_model/feature_12/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0

Hlinear/linear_model/linear_model/linear_model/feature_12/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Flinear/linear_model/linear_model/linear_model/feature_12/Reshape/shapePackFlinear/linear_model/linear_model/linear_model/feature_12/strided_sliceHlinear/linear_model/linear_model/linear_model/feature_12/Reshape/shape/1*
_output_shapes
:*
T0*
N

@linear/linear_model/linear_model/linear_model/feature_12/ReshapeReshape@linear/linear_model/linear_model/linear_model/feature_12/truedivFlinear/linear_model/linear_model/linear_model/feature_12/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0
Ѓ
5linear/linear_model/feature_12/weights/ReadVariableOpReadVariableOp-linear/linear_model/feature_12/weights/part_0*
dtype0*
_output_shapes

:

&linear/linear_model/feature_12/weightsIdentity5linear/linear_model/feature_12/weights/ReadVariableOp*
_output_shapes

:*
T0
ы
Elinear/linear_model/linear_model/linear_model/feature_12/weighted_sumMatMul@linear/linear_model/linear_model/linear_model/feature_12/Reshape&linear/linear_model/feature_12/weights*
T0*'
_output_shapes
:џџџџџџџџџ

>linear/linear_model/linear_model/linear_model/feature_13/sub/yConst*
_output_shapes
: *
valueB
 *ЯШDD*
dtype0
г
<linear/linear_model/linear_model/linear_model/feature_13/subSubParseExample/ParseExample:21>linear/linear_model/linear_model/linear_model/feature_13/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

Blinear/linear_model/linear_model/linear_model/feature_13/truediv/yConst*
valueB
 *hЬ\D*
dtype0*
_output_shapes
: 
џ
@linear/linear_model/linear_model/linear_model/feature_13/truedivRealDiv<linear/linear_model/linear_model/linear_model/feature_13/subBlinear/linear_model/linear_model/linear_model/feature_13/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
>linear/linear_model/linear_model/linear_model/feature_13/ShapeShape@linear/linear_model/linear_model/linear_model/feature_13/truediv*
T0*
_output_shapes
:

Llinear/linear_model/linear_model/linear_model/feature_13/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_13/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Nlinear/linear_model/linear_model/linear_model/feature_13/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ъ
Flinear/linear_model/linear_model/linear_model/feature_13/strided_sliceStridedSlice>linear/linear_model/linear_model/linear_model/feature_13/ShapeLlinear/linear_model/linear_model/linear_model/feature_13/strided_slice/stackNlinear/linear_model/linear_model/linear_model/feature_13/strided_slice/stack_1Nlinear/linear_model/linear_model/linear_model/feature_13/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0

Hlinear/linear_model/linear_model/linear_model/feature_13/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Flinear/linear_model/linear_model/linear_model/feature_13/Reshape/shapePackFlinear/linear_model/linear_model/linear_model/feature_13/strided_sliceHlinear/linear_model/linear_model/linear_model/feature_13/Reshape/shape/1*
_output_shapes
:*
T0*
N

@linear/linear_model/linear_model/linear_model/feature_13/ReshapeReshape@linear/linear_model/linear_model/linear_model/feature_13/truedivFlinear/linear_model/linear_model/linear_model/feature_13/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
5linear/linear_model/feature_13/weights/ReadVariableOpReadVariableOp-linear/linear_model/feature_13/weights/part_0*
dtype0*
_output_shapes

:

&linear/linear_model/feature_13/weightsIdentity5linear/linear_model/feature_13/weights/ReadVariableOp*
_output_shapes

:*
T0
ы
Elinear/linear_model/linear_model/linear_model/feature_13/weighted_sumMatMul@linear/linear_model/linear_model/linear_model/feature_13/Reshape&linear/linear_model/feature_13/weights*'
_output_shapes
:џџџџџџџџџ*
T0
И
Ilinear/linear_model/linear_model/linear_model/feature_14_embedding/lookupStringToHashBucketFastParseExample/ParseExample:6*#
_output_shapes
:џџџџџџџџџ*
num_bucketsm
Е
klinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
Д
jlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

elinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SliceSliceParseExample/ParseExample:12klinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice/beginjlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Џ
elinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
л
dlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/ProdProdelinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Sliceelinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Const*
T0	*
_output_shapes
: 
В
plinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
Џ
mlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0
З
hlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:12plinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2/indicesmlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
ь
flinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Cast/xPackdlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Prodhlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N
Ю
mlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExampleParseExample/ParseExample:12flinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ћ
vlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseReshape/IdentityIdentityIlinear/linear_model/linear_model/linear_model/feature_14_embedding/lookup*#
_output_shapes
:џџџџџџџџџ*
T0	
А
nlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 

llinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GreaterEqualGreaterEqualvlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseReshape/Identitynlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:џџџџџџџџџ

elinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/WhereWherellinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
Р
mlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
і
glinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/ReshapeReshapeelinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Wheremlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Reshape/shape*#
_output_shapes
:џџџџџџџџџ*
T0	
Б
olinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

jlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2_1GatherV2mlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseReshapeglinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Reshapeolinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2_1/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	
Б
olinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

jlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2_2GatherV2vlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseReshape/Identityglinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Reshapeolinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:џџџџџџџџџ*
Taxis0

hlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/IdentityIdentityolinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
Л
ylinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Л
linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsjlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2_1jlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/GatherV2_2hlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Identityylinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
н
linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
п
linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
п
linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
У
linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicelinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowslinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stacklinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0	*
shrink_axis_mask
Щ
|linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/CastCastlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
б
~linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/UniqueUniquelinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0	
І
linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Flinear/linear_model/feature_14_embedding/embedding_weights/part_0/read~linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/Uniquelinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*
Tindices0	*
Tparams0*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*'
_output_shapes
:џџџџџџџџџ
л
linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitylinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:џџџџџџџџџ
г
wlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparseSparseSegmentMeanlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identitylinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/Unique:1|linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
Р
olinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:
Ѓ
ilinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Reshape_1Reshapelinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2olinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ

elinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/ShapeShapewlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
Н
slinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
П
ulinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
П
ulinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

mlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/strided_sliceStridedSliceelinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Shapeslinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/strided_slice/stackulinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_1ulinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
Љ
glinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
ѓ
elinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/stackPackglinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/stack/0mlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/strided_slice*
N*
_output_shapes
:*
T0
љ
dlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/TileTileilinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Reshape_1elinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/stack*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ђ
jlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/zeros_like	ZerosLikewlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
ц
_linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weightsSelectdlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Tilejlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/zeros_likewlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
Р
flinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Cast_1CastParseExample/ParseExample:12*

SrcT0	*
_output_shapes
:*

DstT0
З
mlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
Ж
llinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ч
glinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_1Sliceflinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Cast_1mlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_1/beginllinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
і
glinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Shape_1Shape_linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights*
T0*
_output_shapes
:
З
mlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
П
llinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ш
glinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_2Sliceglinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Shape_1mlinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_2/beginllinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
­
klinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
п
flinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/concatConcatV2glinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_1glinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Slice_2klinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0
я
ilinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Reshape_2Reshape_linear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weightsflinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/concat*'
_output_shapes
:џџџџџџџџџ*
T0
с
Hlinear/linear_model/linear_model/linear_model/feature_14_embedding/ShapeShapeilinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Reshape_2*
_output_shapes
:*
T0
 
Vlinear/linear_model/linear_model/linear_model/feature_14_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ђ
Xlinear/linear_model/linear_model/linear_model/feature_14_embedding/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ђ
Xlinear/linear_model/linear_model/linear_model/feature_14_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ќ
Plinear/linear_model/linear_model/linear_model/feature_14_embedding/strided_sliceStridedSliceHlinear/linear_model/linear_model/linear_model/feature_14_embedding/ShapeVlinear/linear_model/linear_model/linear_model/feature_14_embedding/strided_slice/stackXlinear/linear_model/linear_model/linear_model/feature_14_embedding/strided_slice/stack_1Xlinear/linear_model/linear_model/linear_model/feature_14_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0

Rlinear/linear_model/linear_model/linear_model/feature_14_embedding/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
Ќ
Plinear/linear_model/linear_model/linear_model/feature_14_embedding/Reshape/shapePackPlinear/linear_model/linear_model/linear_model/feature_14_embedding/strided_sliceRlinear/linear_model/linear_model/linear_model/feature_14_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Ф
Jlinear/linear_model/linear_model/linear_model/feature_14_embedding/ReshapeReshapeilinear/linear_model/linear_model/linear_model/feature_14_embedding/feature_14_embedding_weights/Reshape_2Plinear/linear_model/linear_model/linear_model/feature_14_embedding/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
З
?linear/linear_model/feature_14_embedding/weights/ReadVariableOpReadVariableOp7linear/linear_model/feature_14_embedding/weights/part_0*
dtype0*
_output_shapes

:
І
0linear/linear_model/feature_14_embedding/weightsIdentity?linear/linear_model/feature_14_embedding/weights/ReadVariableOp*
_output_shapes

:*
T0

Olinear/linear_model/linear_model/linear_model/feature_14_embedding/weighted_sumMatMulJlinear/linear_model/linear_model/linear_model/feature_14_embedding/Reshape0linear/linear_model/feature_14_embedding/weights*
T0*'
_output_shapes
:џџџџџџџџџ

>linear/linear_model/linear_model/linear_model/feature_15/sub/yConst*
valueB
 *c@*
dtype0*
_output_shapes
: 
г
<linear/linear_model/linear_model/linear_model/feature_15/subSubParseExample/ParseExample:22>linear/linear_model/linear_model/linear_model/feature_15/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

Blinear/linear_model/linear_model/linear_model/feature_15/truediv/yConst*
valueB
 *[8A*
dtype0*
_output_shapes
: 
џ
@linear/linear_model/linear_model/linear_model/feature_15/truedivRealDiv<linear/linear_model/linear_model/linear_model/feature_15/subBlinear/linear_model/linear_model/linear_model/feature_15/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
>linear/linear_model/linear_model/linear_model/feature_15/ShapeShape@linear/linear_model/linear_model/linear_model/feature_15/truediv*
T0*
_output_shapes
:

Llinear/linear_model/linear_model/linear_model/feature_15/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_15/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_15/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ъ
Flinear/linear_model/linear_model/linear_model/feature_15/strided_sliceStridedSlice>linear/linear_model/linear_model/linear_model/feature_15/ShapeLlinear/linear_model/linear_model/linear_model/feature_15/strided_slice/stackNlinear/linear_model/linear_model/linear_model/feature_15/strided_slice/stack_1Nlinear/linear_model/linear_model/linear_model/feature_15/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0

Hlinear/linear_model/linear_model/linear_model/feature_15/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0

Flinear/linear_model/linear_model/linear_model/feature_15/Reshape/shapePackFlinear/linear_model/linear_model/linear_model/feature_15/strided_sliceHlinear/linear_model/linear_model/linear_model/feature_15/Reshape/shape/1*
T0*
N*
_output_shapes
:

@linear/linear_model/linear_model/linear_model/feature_15/ReshapeReshape@linear/linear_model/linear_model/linear_model/feature_15/truedivFlinear/linear_model/linear_model/linear_model/feature_15/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
5linear/linear_model/feature_15/weights/ReadVariableOpReadVariableOp-linear/linear_model/feature_15/weights/part_0*
dtype0*
_output_shapes

:

&linear/linear_model/feature_15/weightsIdentity5linear/linear_model/feature_15/weights/ReadVariableOp*
_output_shapes

:*
T0
ы
Elinear/linear_model/linear_model/linear_model/feature_15/weighted_sumMatMul@linear/linear_model/linear_model/linear_model/feature_15/Reshape&linear/linear_model/feature_15/weights*
T0*'
_output_shapes
:џџџџџџџџџ

>linear/linear_model/linear_model/linear_model/feature_16/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *EрA
г
<linear/linear_model/linear_model/linear_model/feature_16/subSubParseExample/ParseExample:23>linear/linear_model/linear_model/linear_model/feature_16/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

Blinear/linear_model/linear_model/linear_model/feature_16/truediv/yConst*
valueB
 *нгB*
dtype0*
_output_shapes
: 
џ
@linear/linear_model/linear_model/linear_model/feature_16/truedivRealDiv<linear/linear_model/linear_model/linear_model/feature_16/subBlinear/linear_model/linear_model/linear_model/feature_16/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
>linear/linear_model/linear_model/linear_model/feature_16/ShapeShape@linear/linear_model/linear_model/linear_model/feature_16/truediv*
_output_shapes
:*
T0

Llinear/linear_model/linear_model/linear_model/feature_16/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_16/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_16/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ъ
Flinear/linear_model/linear_model/linear_model/feature_16/strided_sliceStridedSlice>linear/linear_model/linear_model/linear_model/feature_16/ShapeLlinear/linear_model/linear_model/linear_model/feature_16/strided_slice/stackNlinear/linear_model/linear_model/linear_model/feature_16/strided_slice/stack_1Nlinear/linear_model/linear_model/linear_model/feature_16/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 

Hlinear/linear_model/linear_model/linear_model/feature_16/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Flinear/linear_model/linear_model/linear_model/feature_16/Reshape/shapePackFlinear/linear_model/linear_model/linear_model/feature_16/strided_sliceHlinear/linear_model/linear_model/linear_model/feature_16/Reshape/shape/1*
T0*
N*
_output_shapes
:

@linear/linear_model/linear_model/linear_model/feature_16/ReshapeReshape@linear/linear_model/linear_model/linear_model/feature_16/truedivFlinear/linear_model/linear_model/linear_model/feature_16/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0
Ѓ
5linear/linear_model/feature_16/weights/ReadVariableOpReadVariableOp-linear/linear_model/feature_16/weights/part_0*
dtype0*
_output_shapes

:

&linear/linear_model/feature_16/weightsIdentity5linear/linear_model/feature_16/weights/ReadVariableOp*
T0*
_output_shapes

:
ы
Elinear/linear_model/linear_model/linear_model/feature_16/weighted_sumMatMul@linear/linear_model/linear_model/linear_model/feature_16/Reshape&linear/linear_model/feature_16/weights*
T0*'
_output_shapes
:џџџџџџџџџ

>linear/linear_model/linear_model/linear_model/feature_17/sub/yConst*
_output_shapes
: *
valueB
 *ВSA*
dtype0
г
<linear/linear_model/linear_model/linear_model/feature_17/subSubParseExample/ParseExample:24>linear/linear_model/linear_model/linear_model/feature_17/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

Blinear/linear_model/linear_model/linear_model/feature_17/truediv/yConst*
_output_shapes
: *
valueB
 *дB*
dtype0
џ
@linear/linear_model/linear_model/linear_model/feature_17/truedivRealDiv<linear/linear_model/linear_model/linear_model/feature_17/subBlinear/linear_model/linear_model/linear_model/feature_17/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ў
>linear/linear_model/linear_model/linear_model/feature_17/ShapeShape@linear/linear_model/linear_model/linear_model/feature_17/truediv*
_output_shapes
:*
T0

Llinear/linear_model/linear_model/linear_model/feature_17/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_17/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Nlinear/linear_model/linear_model/linear_model/feature_17/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ъ
Flinear/linear_model/linear_model/linear_model/feature_17/strided_sliceStridedSlice>linear/linear_model/linear_model/linear_model/feature_17/ShapeLlinear/linear_model/linear_model/linear_model/feature_17/strided_slice/stackNlinear/linear_model/linear_model/linear_model/feature_17/strided_slice/stack_1Nlinear/linear_model/linear_model/linear_model/feature_17/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Hlinear/linear_model/linear_model/linear_model/feature_17/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Flinear/linear_model/linear_model/linear_model/feature_17/Reshape/shapePackFlinear/linear_model/linear_model/linear_model/feature_17/strided_sliceHlinear/linear_model/linear_model/linear_model/feature_17/Reshape/shape/1*
T0*
N*
_output_shapes
:

@linear/linear_model/linear_model/linear_model/feature_17/ReshapeReshape@linear/linear_model/linear_model/linear_model/feature_17/truedivFlinear/linear_model/linear_model/linear_model/feature_17/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
5linear/linear_model/feature_17/weights/ReadVariableOpReadVariableOp-linear/linear_model/feature_17/weights/part_0*
dtype0*
_output_shapes

:

&linear/linear_model/feature_17/weightsIdentity5linear/linear_model/feature_17/weights/ReadVariableOp*
_output_shapes

:*
T0
ы
Elinear/linear_model/linear_model/linear_model/feature_17/weighted_sumMatMul@linear/linear_model/linear_model/linear_model/feature_17/Reshape&linear/linear_model/feature_17/weights*
T0*'
_output_shapes
:џџџџџџџџџ
И
Ilinear/linear_model/linear_model/linear_model/feature_18_embedding/lookupStringToHashBucketFastParseExample/ParseExample:7*#
_output_shapes
:џџџџџџџџџ*
num_buckets 
Е
klinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
Д
jlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

elinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SliceSliceParseExample/ParseExample:13klinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice/beginjlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Џ
elinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
л
dlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/ProdProdelinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Sliceelinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Const*
T0	*
_output_shapes
: 
В
plinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Џ
mlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
З
hlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:13plinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2/indicesmlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2/axis*
Tindices0*
Tparams0	*
_output_shapes
: *
Taxis0
ь
flinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Cast/xPackdlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Prodhlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
а
mlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:1ParseExample/ParseExample:13flinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ћ
vlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseReshape/IdentityIdentityIlinear/linear_model/linear_model/linear_model/feature_18_embedding/lookup*#
_output_shapes
:џџџџџџџџџ*
T0	
А
nlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B	 R *
dtype0	

llinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GreaterEqualGreaterEqualvlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseReshape/Identitynlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:џџџџџџџџџ

elinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/WhereWherellinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
Р
mlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
і
glinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/ReshapeReshapeelinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Wheremlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:џџџџџџџџџ
Б
olinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

jlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2_1GatherV2mlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseReshapeglinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Reshapeolinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:џџџџџџџџџ
Б
olinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

jlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2_2GatherV2vlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseReshape/Identityglinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Reshapeolinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:џџџџџџџџџ*
Taxis0

hlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/IdentityIdentityolinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
Л
ylinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Л
linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsjlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2_1jlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/GatherV2_2hlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Identityylinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
н
linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
п
linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
п
linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
У
linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicelinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowslinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stacklinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0	*
shrink_axis_mask*

begin_mask
Щ
|linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/CastCastlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
б
~linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/UniqueUniquelinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
І
linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Flinear/linear_model/feature_18_embedding/embedding_weights/part_0/read~linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/Uniquelinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*
Tindices0	*
Tparams0*T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*'
_output_shapes
:џџџџџџџџџ
л
linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitylinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:џџџџџџџџџ*
T0
г
wlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparseSparseSegmentMeanlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identitylinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/Unique:1|linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
Р
olinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:
Ѓ
ilinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Reshape_1Reshapelinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2olinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ

elinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/ShapeShapewlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0
Н
slinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
П
ulinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
П
ulinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

mlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/strided_sliceStridedSliceelinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Shapeslinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/strided_slice/stackulinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_1ulinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
Љ
glinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
ѓ
elinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/stackPackglinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/stack/0mlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/strided_slice*
_output_shapes
:*
T0*
N
љ
dlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/TileTileilinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Reshape_1elinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/stack*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ђ
jlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/zeros_like	ZerosLikewlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
ц
_linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weightsSelectdlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Tilejlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/zeros_likewlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
Р
flinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Cast_1CastParseExample/ParseExample:13*
_output_shapes
:*

DstT0*

SrcT0	
З
mlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
Ж
llinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ч
glinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_1Sliceflinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Cast_1mlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_1/beginllinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
і
glinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Shape_1Shape_linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights*
T0*
_output_shapes
:
З
mlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
П
llinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ш
glinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_2Sliceglinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Shape_1mlinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_2/beginllinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
­
klinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
п
flinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/concatConcatV2glinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_1glinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Slice_2klinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0
я
ilinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Reshape_2Reshape_linear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weightsflinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/concat*
T0*'
_output_shapes
:џџџџџџџџџ
с
Hlinear/linear_model/linear_model/linear_model/feature_18_embedding/ShapeShapeilinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Reshape_2*
T0*
_output_shapes
:
 
Vlinear/linear_model/linear_model/linear_model/feature_18_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ђ
Xlinear/linear_model/linear_model/linear_model/feature_18_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Ђ
Xlinear/linear_model/linear_model/linear_model/feature_18_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ќ
Plinear/linear_model/linear_model/linear_model/feature_18_embedding/strided_sliceStridedSliceHlinear/linear_model/linear_model/linear_model/feature_18_embedding/ShapeVlinear/linear_model/linear_model/linear_model/feature_18_embedding/strided_slice/stackXlinear/linear_model/linear_model/linear_model/feature_18_embedding/strided_slice/stack_1Xlinear/linear_model/linear_model/linear_model/feature_18_embedding/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 

Rlinear/linear_model/linear_model/linear_model/feature_18_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ќ
Plinear/linear_model/linear_model/linear_model/feature_18_embedding/Reshape/shapePackPlinear/linear_model/linear_model/linear_model/feature_18_embedding/strided_sliceRlinear/linear_model/linear_model/linear_model/feature_18_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Ф
Jlinear/linear_model/linear_model/linear_model/feature_18_embedding/ReshapeReshapeilinear/linear_model/linear_model/linear_model/feature_18_embedding/feature_18_embedding_weights/Reshape_2Plinear/linear_model/linear_model/linear_model/feature_18_embedding/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0
З
?linear/linear_model/feature_18_embedding/weights/ReadVariableOpReadVariableOp7linear/linear_model/feature_18_embedding/weights/part_0*
_output_shapes

:*
dtype0
І
0linear/linear_model/feature_18_embedding/weightsIdentity?linear/linear_model/feature_18_embedding/weights/ReadVariableOp*
_output_shapes

:*
T0

Olinear/linear_model/linear_model/linear_model/feature_18_embedding/weighted_sumMatMulJlinear/linear_model/linear_model/linear_model/feature_18_embedding/Reshape0linear/linear_model/feature_18_embedding/weights*
T0*'
_output_shapes
:џџџџџџџџџ
И
Hlinear/linear_model/linear_model/linear_model/feature_2_embedding/lookupStringToHashBucketFastParseExample/ParseExample:8*#
_output_shapes
:џџџџџџџџџ*
num_buckets
Г
ilinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
В
hlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

clinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SliceSliceParseExample/ParseExample:14ilinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice/beginhlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
­
clinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
е
blinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/ProdProdclinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Sliceclinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Const*
T0	*
_output_shapes
: 
А
nlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
­
klinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Б
flinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:14nlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2/indicesklinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2/axis*
Tindices0*
Tparams0	*
_output_shapes
: *
Taxis0
ц
dlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Cast/xPackblinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Prodflinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
Ь
klinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:2ParseExample/ParseExample:14dlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ј
tlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseReshape/IdentityIdentityHlinear/linear_model/linear_model/linear_model/feature_2_embedding/lookup*#
_output_shapes
:џџџџџџџџџ*
T0	
Ў
llinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 

jlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GreaterEqualGreaterEqualtlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseReshape/Identityllinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:џџџџџџџџџ

clinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/WhereWherejlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
О
klinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
№
elinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/ReshapeReshapeclinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Whereklinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:џџџџџџџџџ
Џ
mlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

hlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2_1GatherV2klinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseReshapeelinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Reshapemlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2_1/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	
Џ
mlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 

hlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2_2GatherV2tlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseReshape/Identityelinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Reshapemlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:џџџџџџџџџ

flinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/IdentityIdentitymlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
Й
wlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Б
linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowshlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2_1hlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/GatherV2_2flinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Identitywlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
л
linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
н
linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
н
linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Й
linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicelinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowslinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stacklinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ*
T0	*
Index0*
shrink_axis_mask
Х
zlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/CastCastlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
Э
|linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/UniqueUniquelinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѓ
linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Elinear/linear_model/feature_2_embedding/embedding_weights/part_0/read|linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/Uniquelinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0
з
linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitylinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:џџџџџџџџџ*
T0
Ъ
ulinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparseSparseSegmentMeanlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity~linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/Unique:1zlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
О
mlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

glinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Reshape_1Reshapelinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2mlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ

clinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/ShapeShapeulinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
Л
qlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
Н
slinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Н
slinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

klinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/strided_sliceStridedSliceclinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Shapeqlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/strided_slice/stackslinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_1slinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
Ї
elinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
э
clinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/stackPackelinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/stack/0klinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ѓ
blinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/TileTileglinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Reshape_1clinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/stack*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

hlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/zeros_like	ZerosLikeulinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
о
]linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weightsSelectblinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Tilehlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/zeros_likeulinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:џџџџџџџџџ*
T0
О
dlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Cast_1CastParseExample/ParseExample:14*

SrcT0	*
_output_shapes
:*

DstT0
Е
klinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
Д
jlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
п
elinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_1Slicedlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Cast_1klinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_1/beginjlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
ђ
elinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Shape_1Shape]linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights*
_output_shapes
:*
T0
Е
klinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
Н
jlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
р
elinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_2Sliceelinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Shape_1klinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_2/beginjlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ћ
ilinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
з
dlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/concatConcatV2elinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_1elinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Slice_2ilinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
щ
glinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Reshape_2Reshape]linear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weightsdlinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/concat*'
_output_shapes
:џџџџџџџџџ*
T0
о
Glinear/linear_model/linear_model/linear_model/feature_2_embedding/ShapeShapeglinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Reshape_2*
_output_shapes
:*
T0

Ulinear/linear_model/linear_model/linear_model/feature_2_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ё
Wlinear/linear_model/linear_model/linear_model/feature_2_embedding/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ё
Wlinear/linear_model/linear_model/linear_model/feature_2_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ї
Olinear/linear_model/linear_model/linear_model/feature_2_embedding/strided_sliceStridedSliceGlinear/linear_model/linear_model/linear_model/feature_2_embedding/ShapeUlinear/linear_model/linear_model/linear_model/feature_2_embedding/strided_slice/stackWlinear/linear_model/linear_model/linear_model/feature_2_embedding/strided_slice/stack_1Wlinear/linear_model/linear_model/linear_model/feature_2_embedding/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask

Qlinear/linear_model/linear_model/linear_model/feature_2_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Љ
Olinear/linear_model/linear_model/linear_model/feature_2_embedding/Reshape/shapePackOlinear/linear_model/linear_model/linear_model/feature_2_embedding/strided_sliceQlinear/linear_model/linear_model/linear_model/feature_2_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Р
Ilinear/linear_model/linear_model/linear_model/feature_2_embedding/ReshapeReshapeglinear/linear_model/linear_model/linear_model/feature_2_embedding/feature_2_embedding_weights/Reshape_2Olinear/linear_model/linear_model/linear_model/feature_2_embedding/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Е
>linear/linear_model/feature_2_embedding/weights/ReadVariableOpReadVariableOp6linear/linear_model/feature_2_embedding/weights/part_0*
dtype0*
_output_shapes

:
Є
/linear/linear_model/feature_2_embedding/weightsIdentity>linear/linear_model/feature_2_embedding/weights/ReadVariableOp*
_output_shapes

:*
T0

Nlinear/linear_model/linear_model/linear_model/feature_2_embedding/weighted_sumMatMulIlinear/linear_model/linear_model/linear_model/feature_2_embedding/Reshape/linear/linear_model/feature_2_embedding/weights*'
_output_shapes
:џџџџџџџџџ*
T0

=linear/linear_model/linear_model/linear_model/feature_3/sub/yConst*
valueB
 */QD*
dtype0*
_output_shapes
: 
б
;linear/linear_model/linear_model/linear_model/feature_3/subSubParseExample/ParseExample:25=linear/linear_model/linear_model/linear_model/feature_3/sub/y*'
_output_shapes
:џџџџџџџџџ*
T0

Alinear/linear_model/linear_model/linear_model/feature_3/truediv/yConst*
valueB
 *O ИD*
dtype0*
_output_shapes
: 
ќ
?linear/linear_model/linear_model/linear_model/feature_3/truedivRealDiv;linear/linear_model/linear_model/linear_model/feature_3/subAlinear/linear_model/linear_model/linear_model/feature_3/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
=linear/linear_model/linear_model/linear_model/feature_3/ShapeShape?linear/linear_model/linear_model/linear_model/feature_3/truediv*
_output_shapes
:*
T0

Klinear/linear_model/linear_model/linear_model/feature_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Mlinear/linear_model/linear_model/linear_model/feature_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Mlinear/linear_model/linear_model/linear_model/feature_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Х
Elinear/linear_model/linear_model/linear_model/feature_3/strided_sliceStridedSlice=linear/linear_model/linear_model/linear_model/feature_3/ShapeKlinear/linear_model/linear_model/linear_model/feature_3/strided_slice/stackMlinear/linear_model/linear_model/linear_model/feature_3/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/feature_3/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Glinear/linear_model/linear_model/linear_model/feature_3/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Elinear/linear_model/linear_model/linear_model/feature_3/Reshape/shapePackElinear/linear_model/linear_model/linear_model/feature_3/strided_sliceGlinear/linear_model/linear_model/linear_model/feature_3/Reshape/shape/1*
T0*
N*
_output_shapes
:

?linear/linear_model/linear_model/linear_model/feature_3/ReshapeReshape?linear/linear_model/linear_model/linear_model/feature_3/truedivElinear/linear_model/linear_model/linear_model/feature_3/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
4linear/linear_model/feature_3/weights/ReadVariableOpReadVariableOp,linear/linear_model/feature_3/weights/part_0*
dtype0*
_output_shapes

:

%linear/linear_model/feature_3/weightsIdentity4linear/linear_model/feature_3/weights/ReadVariableOp*
_output_shapes

:*
T0
ш
Dlinear/linear_model/linear_model/linear_model/feature_3/weighted_sumMatMul?linear/linear_model/linear_model/linear_model/feature_3/Reshape%linear/linear_model/feature_3/weights*
T0*'
_output_shapes
:џџџџџџџџџ

=linear/linear_model/linear_model/linear_model/feature_4/sub/yConst*
valueB
 *dDB*
dtype0*
_output_shapes
: 
б
;linear/linear_model/linear_model/linear_model/feature_4/subSubParseExample/ParseExample:26=linear/linear_model/linear_model/linear_model/feature_4/sub/y*'
_output_shapes
:џџџџџџџџџ*
T0

Alinear/linear_model/linear_model/linear_model/feature_4/truediv/yConst*
valueB
 *§WпB*
dtype0*
_output_shapes
: 
ќ
?linear/linear_model/linear_model/linear_model/feature_4/truedivRealDiv;linear/linear_model/linear_model/linear_model/feature_4/subAlinear/linear_model/linear_model/linear_model/feature_4/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
=linear/linear_model/linear_model/linear_model/feature_4/ShapeShape?linear/linear_model/linear_model/linear_model/feature_4/truediv*
_output_shapes
:*
T0

Klinear/linear_model/linear_model/linear_model/feature_4/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Mlinear/linear_model/linear_model/linear_model/feature_4/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Mlinear/linear_model/linear_model/linear_model/feature_4/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Х
Elinear/linear_model/linear_model/linear_model/feature_4/strided_sliceStridedSlice=linear/linear_model/linear_model/linear_model/feature_4/ShapeKlinear/linear_model/linear_model/linear_model/feature_4/strided_slice/stackMlinear/linear_model/linear_model/linear_model/feature_4/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/feature_4/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0

Glinear/linear_model/linear_model/linear_model/feature_4/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0

Elinear/linear_model/linear_model/linear_model/feature_4/Reshape/shapePackElinear/linear_model/linear_model/linear_model/feature_4/strided_sliceGlinear/linear_model/linear_model/linear_model/feature_4/Reshape/shape/1*
_output_shapes
:*
T0*
N

?linear/linear_model/linear_model/linear_model/feature_4/ReshapeReshape?linear/linear_model/linear_model/linear_model/feature_4/truedivElinear/linear_model/linear_model/linear_model/feature_4/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
4linear/linear_model/feature_4/weights/ReadVariableOpReadVariableOp,linear/linear_model/feature_4/weights/part_0*
dtype0*
_output_shapes

:

%linear/linear_model/feature_4/weightsIdentity4linear/linear_model/feature_4/weights/ReadVariableOp*
T0*
_output_shapes

:
ш
Dlinear/linear_model/linear_model/linear_model/feature_4/weighted_sumMatMul?linear/linear_model/linear_model/linear_model/feature_4/Reshape%linear/linear_model/feature_4/weights*
T0*'
_output_shapes
:џџџџџџџџџ

=linear/linear_model/linear_model/linear_model/feature_5/sub/yConst*
valueB
 *ДvA*
dtype0*
_output_shapes
: 
б
;linear/linear_model/linear_model/linear_model/feature_5/subSubParseExample/ParseExample:27=linear/linear_model/linear_model/linear_model/feature_5/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ

Alinear/linear_model/linear_model/linear_model/feature_5/truediv/yConst*
_output_shapes
: *
valueB
 *Ыя2A*
dtype0
ќ
?linear/linear_model/linear_model/linear_model/feature_5/truedivRealDiv;linear/linear_model/linear_model/linear_model/feature_5/subAlinear/linear_model/linear_model/linear_model/feature_5/truediv/y*'
_output_shapes
:џџџџџџџџџ*
T0
Ќ
=linear/linear_model/linear_model/linear_model/feature_5/ShapeShape?linear/linear_model/linear_model/linear_model/feature_5/truediv*
T0*
_output_shapes
:

Klinear/linear_model/linear_model/linear_model/feature_5/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Mlinear/linear_model/linear_model/linear_model/feature_5/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Mlinear/linear_model/linear_model/linear_model/feature_5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Х
Elinear/linear_model/linear_model/linear_model/feature_5/strided_sliceStridedSlice=linear/linear_model/linear_model/linear_model/feature_5/ShapeKlinear/linear_model/linear_model/linear_model/feature_5/strided_slice/stackMlinear/linear_model/linear_model/linear_model/feature_5/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/feature_5/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 

Glinear/linear_model/linear_model/linear_model/feature_5/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Elinear/linear_model/linear_model/linear_model/feature_5/Reshape/shapePackElinear/linear_model/linear_model/linear_model/feature_5/strided_sliceGlinear/linear_model/linear_model/linear_model/feature_5/Reshape/shape/1*
T0*
N*
_output_shapes
:

?linear/linear_model/linear_model/linear_model/feature_5/ReshapeReshape?linear/linear_model/linear_model/linear_model/feature_5/truedivElinear/linear_model/linear_model/linear_model/feature_5/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
4linear/linear_model/feature_5/weights/ReadVariableOpReadVariableOp,linear/linear_model/feature_5/weights/part_0*
dtype0*
_output_shapes

:

%linear/linear_model/feature_5/weightsIdentity4linear/linear_model/feature_5/weights/ReadVariableOp*
T0*
_output_shapes

:
ш
Dlinear/linear_model/linear_model/linear_model/feature_5/weighted_sumMatMul?linear/linear_model/linear_model/linear_model/feature_5/Reshape%linear/linear_model/feature_5/weights*'
_output_shapes
:џџџџџџџџџ*
T0

=linear/linear_model/linear_model/linear_model/feature_6/sub/yConst*
_output_shapes
: *
valueB
 *jјC*
dtype0
б
;linear/linear_model/linear_model/linear_model/feature_6/subSubParseExample/ParseExample:28=linear/linear_model/linear_model/linear_model/feature_6/sub/y*'
_output_shapes
:џџџџџџџџџ*
T0

Alinear/linear_model/linear_model/linear_model/feature_6/truediv/yConst*
valueB
 *м)D*
dtype0*
_output_shapes
: 
ќ
?linear/linear_model/linear_model/linear_model/feature_6/truedivRealDiv;linear/linear_model/linear_model/linear_model/feature_6/subAlinear/linear_model/linear_model/linear_model/feature_6/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
=linear/linear_model/linear_model/linear_model/feature_6/ShapeShape?linear/linear_model/linear_model/linear_model/feature_6/truediv*
_output_shapes
:*
T0

Klinear/linear_model/linear_model/linear_model/feature_6/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Mlinear/linear_model/linear_model/linear_model/feature_6/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Mlinear/linear_model/linear_model/linear_model/feature_6/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Х
Elinear/linear_model/linear_model/linear_model/feature_6/strided_sliceStridedSlice=linear/linear_model/linear_model/linear_model/feature_6/ShapeKlinear/linear_model/linear_model/linear_model/feature_6/strided_slice/stackMlinear/linear_model/linear_model/linear_model/feature_6/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/feature_6/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Glinear/linear_model/linear_model/linear_model/feature_6/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Elinear/linear_model/linear_model/linear_model/feature_6/Reshape/shapePackElinear/linear_model/linear_model/linear_model/feature_6/strided_sliceGlinear/linear_model/linear_model/linear_model/feature_6/Reshape/shape/1*
_output_shapes
:*
T0*
N

?linear/linear_model/linear_model/linear_model/feature_6/ReshapeReshape?linear/linear_model/linear_model/linear_model/feature_6/truedivElinear/linear_model/linear_model/linear_model/feature_6/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
4linear/linear_model/feature_6/weights/ReadVariableOpReadVariableOp,linear/linear_model/feature_6/weights/part_0*
dtype0*
_output_shapes

:

%linear/linear_model/feature_6/weightsIdentity4linear/linear_model/feature_6/weights/ReadVariableOp*
T0*
_output_shapes

:
ш
Dlinear/linear_model/linear_model/linear_model/feature_6/weighted_sumMatMul?linear/linear_model/linear_model/linear_model/feature_6/Reshape%linear/linear_model/feature_6/weights*
T0*'
_output_shapes
:џџџџџџџџџ
З
Hlinear/linear_model/linear_model/linear_model/feature_7_embedding/lookupStringToHashBucketFastParseExample/ParseExample:9*#
_output_shapes
:џџџџџџџџџ*
num_buckets
Г
ilinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
В
hlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:

clinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SliceSliceParseExample/ParseExample:15ilinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice/beginhlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
­
clinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
е
blinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/ProdProdclinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Sliceclinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Const*
T0	*
_output_shapes
: 
А
nlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
­
klinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
flinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:15nlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2/indicesklinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2/axis*
Tindices0*
Tparams0	*
_output_shapes
: *
Taxis0
ц
dlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Cast/xPackblinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Prodflinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
Ь
klinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:3ParseExample/ParseExample:15dlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ј
tlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseReshape/IdentityIdentityHlinear/linear_model/linear_model/linear_model/feature_7_embedding/lookup*
T0	*#
_output_shapes
:џџџџџџџџџ
Ў
llinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 

jlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GreaterEqualGreaterEqualtlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseReshape/Identityllinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GreaterEqual/y*#
_output_shapes
:џџџџџџџџџ*
T0	

clinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/WhereWherejlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
О
klinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
№
elinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/ReshapeReshapeclinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Whereklinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Reshape/shape*#
_output_shapes
:џџџџџџџџџ*
T0	
Џ
mlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

hlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2_1GatherV2klinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseReshapeelinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Reshapemlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:џџџџџџџџџ
Џ
mlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

hlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2_2GatherV2tlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseReshape/Identityelinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Reshapemlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2_2/axis*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	

flinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/IdentityIdentitymlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
Й
wlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Б
linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowshlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2_1hlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/GatherV2_2flinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Identitywlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
л
linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
н
linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
н
linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Й
linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicelinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowslinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stacklinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0	*
shrink_axis_mask*

begin_mask
Х
zlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/CastCastlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0	
Э
|linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/UniqueUniquelinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѓ
linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*
_output_shapes
: *
value	B : *S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0

linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Elinear/linear_model/feature_7_embedding/embedding_weights/part_0/read|linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/Uniquelinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0
з
linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitylinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:џџџџџџџџџ*
T0
Ъ
ulinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparseSparseSegmentMeanlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity~linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/Unique:1zlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
О
mlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

glinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Reshape_1Reshapelinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2mlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ

clinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/ShapeShapeulinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
Л
qlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
Н
slinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Н
slinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

klinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/strided_sliceStridedSliceclinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Shapeqlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/strided_slice/stackslinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_1slinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
Ї
elinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
э
clinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/stackPackelinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/stack/0klinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ѓ
blinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/TileTileglinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Reshape_1clinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/stack*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0


hlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/zeros_like	ZerosLikeulinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
о
]linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weightsSelectblinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Tilehlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/zeros_likeulinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:џџџџџџџџџ*
T0
О
dlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Cast_1CastParseExample/ParseExample:15*

SrcT0	*
_output_shapes
:*

DstT0
Е
klinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
Д
jlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
п
elinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_1Slicedlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Cast_1klinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_1/beginjlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
ђ
elinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Shape_1Shape]linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights*
_output_shapes
:*
T0
Е
klinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
valueB:*
dtype0
Н
jlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
р
elinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_2Sliceelinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Shape_1klinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_2/beginjlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ћ
ilinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
з
dlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/concatConcatV2elinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_1elinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Slice_2ilinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
щ
glinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Reshape_2Reshape]linear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weightsdlinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/concat*
T0*'
_output_shapes
:џџџџџџџџџ
о
Glinear/linear_model/linear_model/linear_model/feature_7_embedding/ShapeShapeglinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Reshape_2*
_output_shapes
:*
T0

Ulinear/linear_model/linear_model/linear_model/feature_7_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
Ё
Wlinear/linear_model/linear_model/linear_model/feature_7_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ё
Wlinear/linear_model/linear_model/linear_model/feature_7_embedding/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
ї
Olinear/linear_model/linear_model/linear_model/feature_7_embedding/strided_sliceStridedSliceGlinear/linear_model/linear_model/linear_model/feature_7_embedding/ShapeUlinear/linear_model/linear_model/linear_model/feature_7_embedding/strided_slice/stackWlinear/linear_model/linear_model/linear_model/feature_7_embedding/strided_slice/stack_1Wlinear/linear_model/linear_model/linear_model/feature_7_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Qlinear/linear_model/linear_model/linear_model/feature_7_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Љ
Olinear/linear_model/linear_model/linear_model/feature_7_embedding/Reshape/shapePackOlinear/linear_model/linear_model/linear_model/feature_7_embedding/strided_sliceQlinear/linear_model/linear_model/linear_model/feature_7_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Р
Ilinear/linear_model/linear_model/linear_model/feature_7_embedding/ReshapeReshapeglinear/linear_model/linear_model/linear_model/feature_7_embedding/feature_7_embedding_weights/Reshape_2Olinear/linear_model/linear_model/linear_model/feature_7_embedding/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Е
>linear/linear_model/feature_7_embedding/weights/ReadVariableOpReadVariableOp6linear/linear_model/feature_7_embedding/weights/part_0*
dtype0*
_output_shapes

:
Є
/linear/linear_model/feature_7_embedding/weightsIdentity>linear/linear_model/feature_7_embedding/weights/ReadVariableOp*
T0*
_output_shapes

:

Nlinear/linear_model/linear_model/linear_model/feature_7_embedding/weighted_sumMatMulIlinear/linear_model/linear_model/linear_model/feature_7_embedding/Reshape/linear/linear_model/feature_7_embedding/weights*
T0*'
_output_shapes
:џџџџџџџџџ
И
Hlinear/linear_model/linear_model/linear_model/feature_8_embedding/lookupStringToHashBucketFastParseExample/ParseExample:10*#
_output_shapes
:џџџџџџџџџ*
num_buckets
Г
ilinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
В
hlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

clinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SliceSliceParseExample/ParseExample:16ilinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice/beginhlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
­
clinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
е
blinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/ProdProdclinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Sliceclinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Const*
T0	*
_output_shapes
: 
А
nlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
­
klinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
flinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:16nlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2/indicesklinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2/axis*
Tindices0*
Tparams0	*
_output_shapes
: *
Taxis0
ц
dlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Cast/xPackblinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Prodflinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2*
_output_shapes
:*
T0	*
N
Ь
klinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:4ParseExample/ParseExample:16dlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ј
tlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseReshape/IdentityIdentityHlinear/linear_model/linear_model/linear_model/feature_8_embedding/lookup*#
_output_shapes
:џџџџџџџџџ*
T0	
Ў
llinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 

jlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GreaterEqualGreaterEqualtlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseReshape/Identityllinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GreaterEqual/y*#
_output_shapes
:џџџџџџџџџ*
T0	

clinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/WhereWherejlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
О
klinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
№
elinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/ReshapeReshapeclinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Whereklinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Reshape/shape*#
_output_shapes
:џџџџџџџџџ*
T0	
Џ
mlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

hlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2_1GatherV2klinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseReshapeelinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Reshapemlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2_1/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	
Џ
mlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 

hlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2_2GatherV2tlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseReshape/Identityelinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Reshapemlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2_2/axis*
Tparams0	*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	

flinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/IdentityIdentitymlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
Й
wlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Б
linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowshlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2_1hlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/GatherV2_2flinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Identitywlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
T0	
л
linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
н
linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
н
linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
Й
linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicelinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowslinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stacklinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0	
Х
zlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/CastCastlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
Э
|linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/UniqueUniquelinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѓ
linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Elinear/linear_model/feature_8_embedding/embedding_weights/part_0/read|linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/Uniquelinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*
Tindices0	*
Tparams0*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*'
_output_shapes
:џџџџџџџџџ
з
linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitylinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:џџџџџџџџџ
Ъ
ulinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparseSparseSegmentMeanlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity~linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/Unique:1zlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
О
mlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

glinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Reshape_1Reshapelinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2mlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ

clinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/ShapeShapeulinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
Л
qlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
Н
slinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Н
slinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

klinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/strided_sliceStridedSliceclinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Shapeqlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/strided_slice/stackslinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_1slinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
Ї
elinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
э
clinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/stackPackelinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/stack/0klinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ѓ
blinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/TileTileglinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Reshape_1clinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/stack*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0


hlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/zeros_like	ZerosLikeulinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:џџџџџџџџџ*
T0
о
]linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weightsSelectblinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Tilehlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/zeros_likeulinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:џџџџџџџџџ*
T0
О
dlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Cast_1CastParseExample/ParseExample:16*
_output_shapes
:*

DstT0*

SrcT0	
Е
klinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
Д
jlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
п
elinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_1Slicedlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Cast_1klinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_1/beginjlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
ђ
elinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Shape_1Shape]linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights*
T0*
_output_shapes
:
Е
klinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
Н
jlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
р
elinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_2Sliceelinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Shape_1klinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_2/beginjlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ћ
ilinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
з
dlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/concatConcatV2elinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_1elinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Slice_2ilinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
щ
glinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Reshape_2Reshape]linear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weightsdlinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/concat*
T0*'
_output_shapes
:џџџџџџџџџ
о
Glinear/linear_model/linear_model/linear_model/feature_8_embedding/ShapeShapeglinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Reshape_2*
T0*
_output_shapes
:

Ulinear/linear_model/linear_model/linear_model/feature_8_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ё
Wlinear/linear_model/linear_model/linear_model/feature_8_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ё
Wlinear/linear_model/linear_model/linear_model/feature_8_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ї
Olinear/linear_model/linear_model/linear_model/feature_8_embedding/strided_sliceStridedSliceGlinear/linear_model/linear_model/linear_model/feature_8_embedding/ShapeUlinear/linear_model/linear_model/linear_model/feature_8_embedding/strided_slice/stackWlinear/linear_model/linear_model/linear_model/feature_8_embedding/strided_slice/stack_1Wlinear/linear_model/linear_model/linear_model/feature_8_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Qlinear/linear_model/linear_model/linear_model/feature_8_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Љ
Olinear/linear_model/linear_model/linear_model/feature_8_embedding/Reshape/shapePackOlinear/linear_model/linear_model/linear_model/feature_8_embedding/strided_sliceQlinear/linear_model/linear_model/linear_model/feature_8_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Р
Ilinear/linear_model/linear_model/linear_model/feature_8_embedding/ReshapeReshapeglinear/linear_model/linear_model/linear_model/feature_8_embedding/feature_8_embedding_weights/Reshape_2Olinear/linear_model/linear_model/linear_model/feature_8_embedding/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Е
>linear/linear_model/feature_8_embedding/weights/ReadVariableOpReadVariableOp6linear/linear_model/feature_8_embedding/weights/part_0*
dtype0*
_output_shapes

:
Є
/linear/linear_model/feature_8_embedding/weightsIdentity>linear/linear_model/feature_8_embedding/weights/ReadVariableOp*
T0*
_output_shapes

:

Nlinear/linear_model/linear_model/linear_model/feature_8_embedding/weighted_sumMatMulIlinear/linear_model/linear_model/linear_model/feature_8_embedding/Reshape/linear/linear_model/feature_8_embedding/weights*'
_output_shapes
:џџџџџџџџџ*
T0
Й
Hlinear/linear_model/linear_model/linear_model/feature_9_embedding/lookupStringToHashBucketFastParseExample/ParseExample:11*#
_output_shapes
:џџџџџџџџџ*
num_bucketsЅ
Г
ilinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
В
hlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

clinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SliceSliceParseExample/ParseExample:17ilinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice/beginhlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
­
clinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
е
blinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/ProdProdclinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Sliceclinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Const*
T0	*
_output_shapes
: 
А
nlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
­
klinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Б
flinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2GatherV2ParseExample/ParseExample:17nlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2/indicesklinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2/axis*
_output_shapes
: *
Taxis0*
Tindices0*
Tparams0	
ц
dlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Cast/xPackblinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Prodflinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
Ь
klinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExample:5ParseExample/ParseExample:17dlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Cast/x*-
_output_shapes
:џџџџџџџџџ:
ј
tlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseReshape/IdentityIdentityHlinear/linear_model/linear_model/linear_model/feature_9_embedding/lookup*#
_output_shapes
:џџџџџџџџџ*
T0	
Ў
llinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 

jlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GreaterEqualGreaterEqualtlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseReshape/Identityllinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GreaterEqual/y*#
_output_shapes
:џџџџџџџџџ*
T0	

clinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/WhereWherejlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GreaterEqual*'
_output_shapes
:џџџџџџџџџ
О
klinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
№
elinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/ReshapeReshapeclinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Whereklinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:џџџџџџџџџ
Џ
mlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

hlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2_1GatherV2klinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseReshapeelinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Reshapemlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2_1/axis*
Tparams0	*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	
Џ
mlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 

hlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2_2GatherV2tlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseReshape/Identityelinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Reshapemlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2_2/axis*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	

flinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/IdentityIdentitymlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
Й
wlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Б
linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowshlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2_1hlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/GatherV2_2flinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Identitywlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
л
linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
н
linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
н
linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Й
linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicelinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowslinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stacklinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
Index0*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ
Х
zlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/CastCastlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
Э
|linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/UniqueUniquelinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѓ
linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 

linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Elinear/linear_model/feature_9_embedding/embedding_weights/part_0/read|linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/Uniquelinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0
з
linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitylinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:џџџџџџџџџ
Ъ
ulinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparseSparseSegmentMeanlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity~linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/Unique:1zlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
О
mlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

glinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Reshape_1Reshapelinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2mlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ

clinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/ShapeShapeulinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
Л
qlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
Н
slinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Н
slinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

klinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/strided_sliceStridedSliceclinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Shapeqlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/strided_slice/stackslinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_1slinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
Ї
elinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
э
clinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/stackPackelinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/stack/0klinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
ѓ
blinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/TileTileglinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Reshape_1clinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/stack*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0


hlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/zeros_like	ZerosLikeulinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:џџџџџџџџџ*
T0
о
]linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weightsSelectblinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Tilehlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/zeros_likeulinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:џџџџџџџџџ*
T0
О
dlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Cast_1CastParseExample/ParseExample:17*

SrcT0	*
_output_shapes
:*

DstT0
Е
klinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
Д
jlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
п
elinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_1Slicedlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Cast_1klinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_1/beginjlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
ђ
elinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Shape_1Shape]linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights*
T0*
_output_shapes
:
Е
klinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
Н
jlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
р
elinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_2Sliceelinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Shape_1klinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_2/beginjlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ћ
ilinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
з
dlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/concatConcatV2elinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_1elinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Slice_2ilinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
щ
glinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Reshape_2Reshape]linear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weightsdlinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/concat*'
_output_shapes
:џџџџџџџџџ*
T0
о
Glinear/linear_model/linear_model/linear_model/feature_9_embedding/ShapeShapeglinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Reshape_2*
T0*
_output_shapes
:

Ulinear/linear_model/linear_model/linear_model/feature_9_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ё
Wlinear/linear_model/linear_model/linear_model/feature_9_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ё
Wlinear/linear_model/linear_model/linear_model/feature_9_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ї
Olinear/linear_model/linear_model/linear_model/feature_9_embedding/strided_sliceStridedSliceGlinear/linear_model/linear_model/linear_model/feature_9_embedding/ShapeUlinear/linear_model/linear_model/linear_model/feature_9_embedding/strided_slice/stackWlinear/linear_model/linear_model/linear_model/feature_9_embedding/strided_slice/stack_1Wlinear/linear_model/linear_model/linear_model/feature_9_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0

Qlinear/linear_model/linear_model/linear_model/feature_9_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Љ
Olinear/linear_model/linear_model/linear_model/feature_9_embedding/Reshape/shapePackOlinear/linear_model/linear_model/linear_model/feature_9_embedding/strided_sliceQlinear/linear_model/linear_model/linear_model/feature_9_embedding/Reshape/shape/1*
T0*
N*
_output_shapes
:
Р
Ilinear/linear_model/linear_model/linear_model/feature_9_embedding/ReshapeReshapeglinear/linear_model/linear_model/linear_model/feature_9_embedding/feature_9_embedding_weights/Reshape_2Olinear/linear_model/linear_model/linear_model/feature_9_embedding/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
Е
>linear/linear_model/feature_9_embedding/weights/ReadVariableOpReadVariableOp6linear/linear_model/feature_9_embedding/weights/part_0*
dtype0*
_output_shapes

:
Є
/linear/linear_model/feature_9_embedding/weightsIdentity>linear/linear_model/feature_9_embedding/weights/ReadVariableOp*
T0*
_output_shapes

:

Nlinear/linear_model/linear_model/linear_model/feature_9_embedding/weighted_sumMatMulIlinear/linear_model/linear_model/linear_model/feature_9_embedding/Reshape/linear/linear_model/feature_9_embedding/weights*
T0*'
_output_shapes
:џџџџџџџџџ
№

Blinear/linear_model/linear_model/linear_model/weighted_sum_no_biasAddNElinear/linear_model/linear_model/linear_model/feature_10/weighted_sumElinear/linear_model/linear_model/linear_model/feature_11/weighted_sumElinear/linear_model/linear_model/linear_model/feature_12/weighted_sumElinear/linear_model/linear_model/linear_model/feature_13/weighted_sumOlinear/linear_model/linear_model/linear_model/feature_14_embedding/weighted_sumElinear/linear_model/linear_model/linear_model/feature_15/weighted_sumElinear/linear_model/linear_model/linear_model/feature_16/weighted_sumElinear/linear_model/linear_model/linear_model/feature_17/weighted_sumOlinear/linear_model/linear_model/linear_model/feature_18_embedding/weighted_sumNlinear/linear_model/linear_model/linear_model/feature_2_embedding/weighted_sumDlinear/linear_model/linear_model/linear_model/feature_3/weighted_sumDlinear/linear_model/linear_model/linear_model/feature_4/weighted_sumDlinear/linear_model/linear_model/linear_model/feature_5/weighted_sumDlinear/linear_model/linear_model/linear_model/feature_6/weighted_sumNlinear/linear_model/linear_model/linear_model/feature_7_embedding/weighted_sumNlinear/linear_model/linear_model/linear_model/feature_8_embedding/weighted_sumNlinear/linear_model/linear_model/linear_model/feature_9_embedding/weighted_sum*
T0*
N*'
_output_shapes
:џџџџџџџџџ

/linear/linear_model/bias_weights/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:

 linear/linear_model/bias_weightsIdentity/linear/linear_model/bias_weights/ReadVariableOp*
T0*
_output_shapes
:
н
:linear/linear_model/linear_model/linear_model/weighted_sumBiasAddBlinear/linear_model/linear_model/linear_model/weighted_sum_no_bias linear/linear_model/bias_weights*
T0*'
_output_shapes
:џџџџџџџџџ
y
linear/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
d
linear/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
f
linear/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
linear/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
й
linear/strided_sliceStridedSlicelinear/ReadVariableOplinear/strided_slice/stacklinear/strided_slice/stack_1linear/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
\
linear/bias/tagsConst*
valueB Blinear/bias*
dtype0*
_output_shapes
: 
e
linear/biasScalarSummarylinear/bias/tagslinear/strided_slice*
_output_shapes
: *
T0
Ё
3linear/zero_fraction/total_size/Size/ReadVariableOpReadVariableOp-linear/linear_model/feature_10/weights/part_0*
dtype0*
_output_shapes

:
f
$linear/zero_fraction/total_size/SizeConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Ѓ
5linear/zero_fraction/total_size/Size_1/ReadVariableOpReadVariableOp-linear/linear_model/feature_11/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_1Const*
value	B	 R*
dtype0	*
_output_shapes
: 
Ѓ
5linear/zero_fraction/total_size/Size_2/ReadVariableOpReadVariableOp-linear/linear_model/feature_12/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_2Const*
_output_shapes
: *
value	B	 R*
dtype0	
Ѓ
5linear/zero_fraction/total_size/Size_3/ReadVariableOpReadVariableOp-linear/linear_model/feature_13/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_3Const*
value	B	 R*
dtype0	*
_output_shapes
: 
i
&linear/zero_fraction/total_size/Size_4Const*
value
B	 RЅ*
dtype0	*
_output_shapes
: 
­
5linear/zero_fraction/total_size/Size_5/ReadVariableOpReadVariableOp7linear/linear_model/feature_14_embedding/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_5Const*
_output_shapes
: *
value	B	 R*
dtype0	
Ѓ
5linear/zero_fraction/total_size/Size_6/ReadVariableOpReadVariableOp-linear/linear_model/feature_15/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_6Const*
value	B	 R*
dtype0	*
_output_shapes
: 
Ѓ
5linear/zero_fraction/total_size/Size_7/ReadVariableOpReadVariableOp-linear/linear_model/feature_16/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_7Const*
_output_shapes
: *
value	B	 R*
dtype0	
Ѓ
5linear/zero_fraction/total_size/Size_8/ReadVariableOpReadVariableOp-linear/linear_model/feature_17/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_8Const*
value	B	 R*
dtype0	*
_output_shapes
: 
i
&linear/zero_fraction/total_size/Size_9Const*
value
B	 Rр*
dtype0	*
_output_shapes
: 
Ў
6linear/zero_fraction/total_size/Size_10/ReadVariableOpReadVariableOp7linear/linear_model/feature_18_embedding/weights/part_0*
dtype0*
_output_shapes

:
i
'linear/zero_fraction/total_size/Size_10Const*
_output_shapes
: *
value	B	 R*
dtype0	
j
'linear/zero_fraction/total_size/Size_11Const*
value
B	 R*
dtype0	*
_output_shapes
: 
­
6linear/zero_fraction/total_size/Size_12/ReadVariableOpReadVariableOp6linear/linear_model/feature_2_embedding/weights/part_0*
dtype0*
_output_shapes

:
i
'linear/zero_fraction/total_size/Size_12Const*
value	B	 R*
dtype0	*
_output_shapes
: 
Ѓ
6linear/zero_fraction/total_size/Size_13/ReadVariableOpReadVariableOp,linear/linear_model/feature_3/weights/part_0*
dtype0*
_output_shapes

:
i
'linear/zero_fraction/total_size/Size_13Const*
value	B	 R*
dtype0	*
_output_shapes
: 
Ѓ
6linear/zero_fraction/total_size/Size_14/ReadVariableOpReadVariableOp,linear/linear_model/feature_4/weights/part_0*
dtype0*
_output_shapes

:
i
'linear/zero_fraction/total_size/Size_14Const*
dtype0	*
_output_shapes
: *
value	B	 R
Ѓ
6linear/zero_fraction/total_size/Size_15/ReadVariableOpReadVariableOp,linear/linear_model/feature_5/weights/part_0*
dtype0*
_output_shapes

:
i
'linear/zero_fraction/total_size/Size_15Const*
value	B	 R*
dtype0	*
_output_shapes
: 
Ѓ
6linear/zero_fraction/total_size/Size_16/ReadVariableOpReadVariableOp,linear/linear_model/feature_6/weights/part_0*
dtype0*
_output_shapes

:
i
'linear/zero_fraction/total_size/Size_16Const*
_output_shapes
: *
value	B	 R*
dtype0	
i
'linear/zero_fraction/total_size/Size_17Const*
dtype0	*
_output_shapes
: *
value	B	 R
­
6linear/zero_fraction/total_size/Size_18/ReadVariableOpReadVariableOp6linear/linear_model/feature_7_embedding/weights/part_0*
dtype0*
_output_shapes

:
i
'linear/zero_fraction/total_size/Size_18Const*
value	B	 R*
dtype0	*
_output_shapes
: 
j
'linear/zero_fraction/total_size/Size_19Const*
value
B	 RЇ*
dtype0	*
_output_shapes
: 
­
6linear/zero_fraction/total_size/Size_20/ReadVariableOpReadVariableOp6linear/linear_model/feature_8_embedding/weights/part_0*
dtype0*
_output_shapes

:
i
'linear/zero_fraction/total_size/Size_20Const*
value	B	 R*
dtype0	*
_output_shapes
: 
j
'linear/zero_fraction/total_size/Size_21Const*
value
B	 R *
dtype0	*
_output_shapes
: 
­
6linear/zero_fraction/total_size/Size_22/ReadVariableOpReadVariableOp6linear/linear_model/feature_9_embedding/weights/part_0*
dtype0*
_output_shapes

:
i
'linear/zero_fraction/total_size/Size_22Const*
value	B	 R*
dtype0	*
_output_shapes
: 
љ
$linear/zero_fraction/total_size/AddNAddN$linear/zero_fraction/total_size/Size&linear/zero_fraction/total_size/Size_1&linear/zero_fraction/total_size/Size_2&linear/zero_fraction/total_size/Size_3&linear/zero_fraction/total_size/Size_4&linear/zero_fraction/total_size/Size_5&linear/zero_fraction/total_size/Size_6&linear/zero_fraction/total_size/Size_7&linear/zero_fraction/total_size/Size_8&linear/zero_fraction/total_size/Size_9'linear/zero_fraction/total_size/Size_10'linear/zero_fraction/total_size/Size_11'linear/zero_fraction/total_size/Size_12'linear/zero_fraction/total_size/Size_13'linear/zero_fraction/total_size/Size_14'linear/zero_fraction/total_size/Size_15'linear/zero_fraction/total_size/Size_16'linear/zero_fraction/total_size/Size_17'linear/zero_fraction/total_size/Size_18'linear/zero_fraction/total_size/Size_19'linear/zero_fraction/total_size/Size_20'linear/zero_fraction/total_size/Size_21'linear/zero_fraction/total_size/Size_22*
_output_shapes
: *
T0	*
N
g
%linear/zero_fraction/total_zero/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 

%linear/zero_fraction/total_zero/EqualEqual$linear/zero_fraction/total_size/Size%linear/zero_fraction/total_zero/Const*
_output_shapes
: *
T0	
Ќ
1linear/zero_fraction/total_zero/zero_count/SwitchSwitch%linear/zero_fraction/total_zero/Equal%linear/zero_fraction/total_zero/Equal*
T0
*
_output_shapes
: : 

3linear/zero_fraction/total_zero/zero_count/switch_tIdentity3linear/zero_fraction/total_zero/zero_count/Switch:1*
_output_shapes
: *
T0


3linear/zero_fraction/total_zero/zero_count/switch_fIdentity1linear/zero_fraction/total_zero/zero_count/Switch*
T0
*
_output_shapes
: 

2linear/zero_fraction/total_zero/zero_count/pred_idIdentity%linear/zero_fraction/total_zero/Equal*
T0
*
_output_shapes
: 
Ћ
0linear/zero_fraction/total_zero/zero_count/ConstConst4^linear/zero_fraction/total_zero/zero_count/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0
ж
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpReadVariableOpNlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch*
_output_shapes

:*
dtype0
 
Nlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/SwitchSwitch-linear/linear_model/feature_10/weights/part_02linear/zero_fraction/total_zero/zero_count/pred_id*
T0*@
_class6
42loc:@linear/linear_model/feature_10/weights/part_0*
_output_shapes
: : 
Е
=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeConst4^linear/zero_fraction/total_zero/zero_count/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Р
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/yConst4^linear/zero_fraction/total_zero/zero_count/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ѕ
Blinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual	LessEqual=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeDlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
љ
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/SwitchSwitchBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqualBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
Л
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_tIdentityFlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
Й
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_fIdentityDlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch*
_output_shapes
: *
T0

Ж
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_idIdentityBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual*
_output_shapes
: *
T0

п
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zerosConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Л
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqualNotEqual]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:

[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchGlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpElinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id*
T0*Z
_classP
NLloc:@linear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp*(
_output_shapes
::
ц
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/CastCastTlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual*
_output_shapes

:*

DstT0*

SrcT0

ы
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/ConstConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
І
Ylinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_countSumPlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/CastQlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
е
Blinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/CastCastYlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
с
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zerosConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
П
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:

]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchGlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpElinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id*
T0*Z
_classP
NLloc:@linear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp*(
_output_shapes
::
ъ
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/CastCastVlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual*
_output_shapes

:*

DstT0	*

SrcT0

э
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/ConstConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Ќ
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/CastSlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	

Clinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/MergeMerge[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_countBlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast*
_output_shapes
: : *
T0	*
N
ћ
Olinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/subSub=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeClinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
й
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/CastCastOlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Щ
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1Cast=linear/zero_fraction/total_zero/zero_count/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ѕ
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truedivRealDivPlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/CastRlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
У
Alinear/zero_fraction/total_zero/zero_count/zero_fraction/fractionIdentitySlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Ѕ
2linear/zero_fraction/total_zero/zero_count/ToFloatCast9linear/zero_fraction/total_zero/zero_count/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0
љ
9linear/zero_fraction/total_zero/zero_count/ToFloat/SwitchSwitch$linear/zero_fraction/total_size/Size2linear/zero_fraction/total_zero/zero_count/pred_id*
T0	*7
_class-
+)loc:@linear/zero_fraction/total_size/Size*
_output_shapes
: : 
Э
.linear/zero_fraction/total_zero/zero_count/mulMulAlinear/zero_fraction/total_zero/zero_count/zero_fraction/fraction2linear/zero_fraction/total_zero/zero_count/ToFloat*
T0*
_output_shapes
: 
Ч
0linear/zero_fraction/total_zero/zero_count/MergeMerge.linear/zero_fraction/total_zero/zero_count/mul0linear/zero_fraction/total_zero/zero_count/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_1Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ђ
'linear/zero_fraction/total_zero/Equal_1Equal&linear/zero_fraction/total_size/Size_1'linear/zero_fraction/total_zero/Const_1*
_output_shapes
: *
T0	
В
3linear/zero_fraction/total_zero/zero_count_1/SwitchSwitch'linear/zero_fraction/total_zero/Equal_1'linear/zero_fraction/total_zero/Equal_1*
T0
*
_output_shapes
: : 

5linear/zero_fraction/total_zero/zero_count_1/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_1/Switch:1*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_1/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_1/Switch*
T0
*
_output_shapes
: 

4linear/zero_fraction/total_zero/zero_count_1/pred_idIdentity'linear/zero_fraction/total_zero/Equal_1*
T0
*
_output_shapes
: 
Џ
2linear/zero_fraction/total_zero/zero_count_1/ConstConst6^linear/zero_fraction/total_zero/zero_count_1/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
к
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
Є
Plinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/SwitchSwitch-linear/linear_model/feature_11/weights/part_04linear/zero_fraction/total_zero/zero_count_1/pred_id*
T0*@
_class6
42loc:@linear/linear_model/feature_11/weights/part_0*
_output_shapes
: : 
Й
?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_1/switch_f*
_output_shapes
: *
value	B	 R*
dtype0	
Ф
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_1/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ћ
Dlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
џ
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual*
_output_shapes
: : *
T0

П
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

Н
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
К
Glinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual*
_output_shapes
: *
T0

у
Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
С
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:

]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp
ъ
Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual*
_output_shapes

:*

DstT0*

SrcT0

я
Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Ќ
[linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
й
Dlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
х
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Х
Xlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:

_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp
ю
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual*
_output_shapes

:*

DstT0	*

SrcT0

ё
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
В
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	

Elinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Qlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
н
Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Э
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
Ћ
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
Ч
Clinear/zero_fraction/total_zero/zero_count_1/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Љ
4linear/zero_fraction/total_zero/zero_count_1/ToFloatCast;linear/zero_fraction/total_zero/zero_count_1/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	

;linear/zero_fraction/total_zero/zero_count_1/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_14linear/zero_fraction/total_zero/zero_count_1/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_1*
_output_shapes
: : 
г
0linear/zero_fraction/total_zero/zero_count_1/mulMulClinear/zero_fraction/total_zero/zero_count_1/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_1/ToFloat*
T0*
_output_shapes
: 
Э
2linear/zero_fraction/total_zero/zero_count_1/MergeMerge0linear/zero_fraction/total_zero/zero_count_1/mul2linear/zero_fraction/total_zero/zero_count_1/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_2Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ђ
'linear/zero_fraction/total_zero/Equal_2Equal&linear/zero_fraction/total_size/Size_2'linear/zero_fraction/total_zero/Const_2*
T0	*
_output_shapes
: 
В
3linear/zero_fraction/total_zero/zero_count_2/SwitchSwitch'linear/zero_fraction/total_zero/Equal_2'linear/zero_fraction/total_zero/Equal_2*
_output_shapes
: : *
T0


5linear/zero_fraction/total_zero/zero_count_2/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_2/Switch:1*
_output_shapes
: *
T0


5linear/zero_fraction/total_zero/zero_count_2/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_2/Switch*
_output_shapes
: *
T0


4linear/zero_fraction/total_zero/zero_count_2/pred_idIdentity'linear/zero_fraction/total_zero/Equal_2*
T0
*
_output_shapes
: 
Џ
2linear/zero_fraction/total_zero/zero_count_2/ConstConst6^linear/zero_fraction/total_zero/zero_count_2/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0
к
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
Є
Plinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/SwitchSwitch-linear/linear_model/feature_12/weights/part_04linear/zero_fraction/total_zero/zero_count_2/pred_id*
_output_shapes
: : *
T0*@
_class6
42loc:@linear/linear_model/feature_12/weights/part_0
Й
?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_2/switch_f*
dtype0	*
_output_shapes
: *
value	B	 R
Ф
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_2/switch_f*
_output_shapes
: *
valueB	 Rџџџџ*
dtype0	
ћ
Dlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
џ
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual*
_output_shapes
: : *
T0

П
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

Н
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
К
Glinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
у
Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0
С
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

:*
T0

]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp*(
_output_shapes
::*
T0
ъ
Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
я
Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Ќ
[linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
й
Dlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
х
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Х
Xlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

:*
T0

_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp*(
_output_shapes
::
ю
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
ё
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
В
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Elinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Qlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
н
Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Э
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
Ћ
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Ч
Clinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Љ
4linear/zero_fraction/total_zero/zero_count_2/ToFloatCast;linear/zero_fraction/total_zero/zero_count_2/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

;linear/zero_fraction/total_zero/zero_count_2/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_24linear/zero_fraction/total_zero/zero_count_2/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_2*
_output_shapes
: : 
г
0linear/zero_fraction/total_zero/zero_count_2/mulMulClinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_2/ToFloat*
_output_shapes
: *
T0
Э
2linear/zero_fraction/total_zero/zero_count_2/MergeMerge0linear/zero_fraction/total_zero/zero_count_2/mul2linear/zero_fraction/total_zero/zero_count_2/Const*
_output_shapes
: : *
T0*
N
i
'linear/zero_fraction/total_zero/Const_3Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ђ
'linear/zero_fraction/total_zero/Equal_3Equal&linear/zero_fraction/total_size/Size_3'linear/zero_fraction/total_zero/Const_3*
T0	*
_output_shapes
: 
В
3linear/zero_fraction/total_zero/zero_count_3/SwitchSwitch'linear/zero_fraction/total_zero/Equal_3'linear/zero_fraction/total_zero/Equal_3*
T0
*
_output_shapes
: : 

5linear/zero_fraction/total_zero/zero_count_3/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_3/Switch:1*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_3/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_3/Switch*
T0
*
_output_shapes
: 

4linear/zero_fraction/total_zero/zero_count_3/pred_idIdentity'linear/zero_fraction/total_zero/Equal_3*
T0
*
_output_shapes
: 
Џ
2linear/zero_fraction/total_zero/zero_count_3/ConstConst6^linear/zero_fraction/total_zero/zero_count_3/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
к
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
Є
Plinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/SwitchSwitch-linear/linear_model/feature_13/weights/part_04linear/zero_fraction/total_zero/zero_count_3/pred_id*
_output_shapes
: : *
T0*@
_class6
42loc:@linear/linear_model/feature_13/weights/part_0
Й
?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_3/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ф
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_3/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ћ
Dlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
џ
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
П
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

Н
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
К
Glinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
у
Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
С
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:

]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp
ъ
Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
я
Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
Ќ
[linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
й
Dlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
х
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Х
Xlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

:*
T0

_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp*(
_output_shapes
::
ю
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
ё
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
В
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	

Elinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Qlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
н
Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Э
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ћ
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Ч
Clinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Љ
4linear/zero_fraction/total_zero/zero_count_3/ToFloatCast;linear/zero_fraction/total_zero/zero_count_3/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

;linear/zero_fraction/total_zero/zero_count_3/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_34linear/zero_fraction/total_zero/zero_count_3/pred_id*
_output_shapes
: : *
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_3
г
0linear/zero_fraction/total_zero/zero_count_3/mulMulClinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_3/ToFloat*
T0*
_output_shapes
: 
Э
2linear/zero_fraction/total_zero/zero_count_3/MergeMerge0linear/zero_fraction/total_zero/zero_count_3/mul2linear/zero_fraction/total_zero/zero_count_3/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_4Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ђ
'linear/zero_fraction/total_zero/Equal_4Equal&linear/zero_fraction/total_size/Size_4'linear/zero_fraction/total_zero/Const_4*
T0	*
_output_shapes
: 
В
3linear/zero_fraction/total_zero/zero_count_4/SwitchSwitch'linear/zero_fraction/total_zero/Equal_4'linear/zero_fraction/total_zero/Equal_4*
T0
*
_output_shapes
: : 

5linear/zero_fraction/total_zero/zero_count_4/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_4/Switch:1*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_4/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_4/Switch*
T0
*
_output_shapes
: 

4linear/zero_fraction/total_zero/zero_count_4/pred_idIdentity'linear/zero_fraction/total_zero/Equal_4*
_output_shapes
: *
T0

Џ
2linear/zero_fraction/total_zero/zero_count_4/ConstConst6^linear/zero_fraction/total_zero/zero_count_4/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
К
?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_4/switch_f*
value
B	 RЅ*
dtype0	*
_output_shapes
: 
Ф
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_4/switch_f*
dtype0	*
_output_shapes
: *
valueB	 Rџџџџ
ћ
Dlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
џ
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
П
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
Н
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
К
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
у
Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
У
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqualNotEqualalinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:m
ю
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchFlinear/linear_model/feature_14_embedding/embedding_weights/part_0/read4linear/zero_fraction/total_zero/zero_count_4/pred_id*
T0*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*(
_output_shapes
:m:m

_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch_1Switch]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/SwitchGlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id*(
_output_shapes
:m:m*
T0*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0
ъ
Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:m*

DstT0
я
Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Ќ
[linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
й
Dlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
х
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Х
Xlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:m

_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/SwitchGlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id*
T0*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*(
_output_shapes
:m:m
ю
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:m*

DstT0	
ё
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
В
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Elinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	

Qlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
н
Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Э
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
Ћ
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
Ч
Clinear/zero_fraction/total_zero/zero_count_4/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Љ
4linear/zero_fraction/total_zero/zero_count_4/ToFloatCast;linear/zero_fraction/total_zero/zero_count_4/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

;linear/zero_fraction/total_zero/zero_count_4/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_44linear/zero_fraction/total_zero/zero_count_4/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_4*
_output_shapes
: : 
г
0linear/zero_fraction/total_zero/zero_count_4/mulMulClinear/zero_fraction/total_zero/zero_count_4/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_4/ToFloat*
_output_shapes
: *
T0
Э
2linear/zero_fraction/total_zero/zero_count_4/MergeMerge0linear/zero_fraction/total_zero/zero_count_4/mul2linear/zero_fraction/total_zero/zero_count_4/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_5Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ђ
'linear/zero_fraction/total_zero/Equal_5Equal&linear/zero_fraction/total_size/Size_5'linear/zero_fraction/total_zero/Const_5*
T0	*
_output_shapes
: 
В
3linear/zero_fraction/total_zero/zero_count_5/SwitchSwitch'linear/zero_fraction/total_zero/Equal_5'linear/zero_fraction/total_zero/Equal_5*
_output_shapes
: : *
T0


5linear/zero_fraction/total_zero/zero_count_5/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_5/Switch:1*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_5/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_5/Switch*
_output_shapes
: *
T0


4linear/zero_fraction/total_zero/zero_count_5/pred_idIdentity'linear/zero_fraction/total_zero/Equal_5*
T0
*
_output_shapes
: 
Џ
2linear/zero_fraction/total_zero/zero_count_5/ConstConst6^linear/zero_fraction/total_zero/zero_count_5/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
к
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
И
Plinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/SwitchSwitch7linear/linear_model/feature_14_embedding/weights/part_04linear/zero_fraction/total_zero/zero_count_5/pred_id*
T0*J
_class@
><loc:@linear/linear_model/feature_14_embedding/weights/part_0*
_output_shapes
: : 
Й
?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_5/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ф
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_5/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ћ
Dlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
џ
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual*
_output_shapes
: : *
T0

П
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
Н
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch*
_output_shapes
: *
T0

К
Glinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual*
_output_shapes
: *
T0

у
Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
С
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:

]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp*(
_output_shapes
::
ъ
Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual*
_output_shapes

:*

DstT0*

SrcT0

я
Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Ќ
[linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
й
Dlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
х
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Х
Xlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

:*
T0

_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp*(
_output_shapes
::
ю
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual*
_output_shapes

:*

DstT0	*

SrcT0

ё
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f*
_output_shapes
:*
valueB"       *
dtype0
В
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Elinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Cast*
_output_shapes
: : *
T0	*
N

Qlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
н
Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Э
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ћ
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Ч
Clinear/zero_fraction/total_zero/zero_count_5/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Љ
4linear/zero_fraction/total_zero/zero_count_5/ToFloatCast;linear/zero_fraction/total_zero/zero_count_5/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	

;linear/zero_fraction/total_zero/zero_count_5/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_54linear/zero_fraction/total_zero/zero_count_5/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_5*
_output_shapes
: : 
г
0linear/zero_fraction/total_zero/zero_count_5/mulMulClinear/zero_fraction/total_zero/zero_count_5/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_5/ToFloat*
_output_shapes
: *
T0
Э
2linear/zero_fraction/total_zero/zero_count_5/MergeMerge0linear/zero_fraction/total_zero/zero_count_5/mul2linear/zero_fraction/total_zero/zero_count_5/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_6Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ђ
'linear/zero_fraction/total_zero/Equal_6Equal&linear/zero_fraction/total_size/Size_6'linear/zero_fraction/total_zero/Const_6*
_output_shapes
: *
T0	
В
3linear/zero_fraction/total_zero/zero_count_6/SwitchSwitch'linear/zero_fraction/total_zero/Equal_6'linear/zero_fraction/total_zero/Equal_6*
T0
*
_output_shapes
: : 

5linear/zero_fraction/total_zero/zero_count_6/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_6/Switch:1*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_6/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_6/Switch*
T0
*
_output_shapes
: 

4linear/zero_fraction/total_zero/zero_count_6/pred_idIdentity'linear/zero_fraction/total_zero/Equal_6*
T0
*
_output_shapes
: 
Џ
2linear/zero_fraction/total_zero/zero_count_6/ConstConst6^linear/zero_fraction/total_zero/zero_count_6/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
к
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
Є
Plinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/SwitchSwitch-linear/linear_model/feature_15/weights/part_04linear/zero_fraction/total_zero/zero_count_6/pred_id*
T0*@
_class6
42loc:@linear/linear_model/feature_15/weights/part_0*
_output_shapes
: : 
Й
?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_6/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ф
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_6/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ћ
Dlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
џ
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual*
_output_shapes
: : *
T0

П
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

Н
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch*
_output_shapes
: *
T0

К
Glinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
у
Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
С
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:

]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp
ъ
Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
я
Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Ќ
[linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
й
Dlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
х
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Х
Xlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:

_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp*(
_output_shapes
::*
T0
ю
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
ё
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
В
]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Elinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Qlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
н
Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Э
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
Ћ
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
Ч
Clinear/zero_fraction/total_zero/zero_count_6/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Љ
4linear/zero_fraction/total_zero/zero_count_6/ToFloatCast;linear/zero_fraction/total_zero/zero_count_6/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

;linear/zero_fraction/total_zero/zero_count_6/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_64linear/zero_fraction/total_zero/zero_count_6/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_6*
_output_shapes
: : 
г
0linear/zero_fraction/total_zero/zero_count_6/mulMulClinear/zero_fraction/total_zero/zero_count_6/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_6/ToFloat*
T0*
_output_shapes
: 
Э
2linear/zero_fraction/total_zero/zero_count_6/MergeMerge0linear/zero_fraction/total_zero/zero_count_6/mul2linear/zero_fraction/total_zero/zero_count_6/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_7Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ђ
'linear/zero_fraction/total_zero/Equal_7Equal&linear/zero_fraction/total_size/Size_7'linear/zero_fraction/total_zero/Const_7*
_output_shapes
: *
T0	
В
3linear/zero_fraction/total_zero/zero_count_7/SwitchSwitch'linear/zero_fraction/total_zero/Equal_7'linear/zero_fraction/total_zero/Equal_7*
T0
*
_output_shapes
: : 

5linear/zero_fraction/total_zero/zero_count_7/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_7/Switch:1*
_output_shapes
: *
T0


5linear/zero_fraction/total_zero/zero_count_7/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_7/Switch*
_output_shapes
: *
T0


4linear/zero_fraction/total_zero/zero_count_7/pred_idIdentity'linear/zero_fraction/total_zero/Equal_7*
_output_shapes
: *
T0

Џ
2linear/zero_fraction/total_zero/zero_count_7/ConstConst6^linear/zero_fraction/total_zero/zero_count_7/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
к
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/Switch*
_output_shapes

:*
dtype0
Є
Plinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/SwitchSwitch-linear/linear_model/feature_16/weights/part_04linear/zero_fraction/total_zero/zero_count_7/pred_id*
T0*@
_class6
42loc:@linear/linear_model/feature_16/weights/part_0*
_output_shapes
: : 
Й
?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_7/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ф
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_7/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ћ
Dlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
џ
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
П
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

Н
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
К
Glinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
у
Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
С
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

:*
T0

]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp*(
_output_shapes
::
ъ
Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
я
Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t*
_output_shapes
:*
valueB"       *
dtype0
Ќ
[linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
й
Dlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
х
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
Х
Xlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:

_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp*(
_output_shapes
::
ю
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
ё
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
В
]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	

Elinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	

Qlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
н
Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Э
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ћ
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
Ч
Clinear/zero_fraction/total_zero/zero_count_7/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Љ
4linear/zero_fraction/total_zero/zero_count_7/ToFloatCast;linear/zero_fraction/total_zero/zero_count_7/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

;linear/zero_fraction/total_zero/zero_count_7/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_74linear/zero_fraction/total_zero/zero_count_7/pred_id*
_output_shapes
: : *
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_7
г
0linear/zero_fraction/total_zero/zero_count_7/mulMulClinear/zero_fraction/total_zero/zero_count_7/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_7/ToFloat*
_output_shapes
: *
T0
Э
2linear/zero_fraction/total_zero/zero_count_7/MergeMerge0linear/zero_fraction/total_zero/zero_count_7/mul2linear/zero_fraction/total_zero/zero_count_7/Const*
N*
_output_shapes
: : *
T0
i
'linear/zero_fraction/total_zero/Const_8Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ђ
'linear/zero_fraction/total_zero/Equal_8Equal&linear/zero_fraction/total_size/Size_8'linear/zero_fraction/total_zero/Const_8*
T0	*
_output_shapes
: 
В
3linear/zero_fraction/total_zero/zero_count_8/SwitchSwitch'linear/zero_fraction/total_zero/Equal_8'linear/zero_fraction/total_zero/Equal_8*
T0
*
_output_shapes
: : 

5linear/zero_fraction/total_zero/zero_count_8/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_8/Switch:1*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_8/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_8/Switch*
T0
*
_output_shapes
: 

4linear/zero_fraction/total_zero/zero_count_8/pred_idIdentity'linear/zero_fraction/total_zero/Equal_8*
_output_shapes
: *
T0

Џ
2linear/zero_fraction/total_zero/zero_count_8/ConstConst6^linear/zero_fraction/total_zero/zero_count_8/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0
к
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
Є
Plinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp/SwitchSwitch-linear/linear_model/feature_17/weights/part_04linear/zero_fraction/total_zero/zero_count_8/pred_id*
_output_shapes
: : *
T0*@
_class6
42loc:@linear/linear_model/feature_17/weights/part_0
Й
?linear/zero_fraction/total_zero/zero_count_8/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_8/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ф
Flinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_8/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ћ
Dlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_8/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
џ
Flinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
П
Hlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

Н
Hlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
К
Glinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
у
Slinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
С
Vlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

:*
T0

]linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp*(
_output_shapes
::
ъ
Rlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
я
Slinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Ќ
[linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
й
Dlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
х
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Х
Xlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:

_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp*(
_output_shapes
::*
T0
ю
Tlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
ё
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_f*
_output_shapes
:*
valueB"       *
dtype0
В
]linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	

Elinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Qlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_8/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
н
Rlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
Э
Tlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_8/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
Ћ
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
Ч
Clinear/zero_fraction/total_zero/zero_count_8/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Љ
4linear/zero_fraction/total_zero/zero_count_8/ToFloatCast;linear/zero_fraction/total_zero/zero_count_8/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	

;linear/zero_fraction/total_zero/zero_count_8/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_84linear/zero_fraction/total_zero/zero_count_8/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_8*
_output_shapes
: : 
г
0linear/zero_fraction/total_zero/zero_count_8/mulMulClinear/zero_fraction/total_zero/zero_count_8/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_8/ToFloat*
_output_shapes
: *
T0
Э
2linear/zero_fraction/total_zero/zero_count_8/MergeMerge0linear/zero_fraction/total_zero/zero_count_8/mul2linear/zero_fraction/total_zero/zero_count_8/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_9Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ђ
'linear/zero_fraction/total_zero/Equal_9Equal&linear/zero_fraction/total_size/Size_9'linear/zero_fraction/total_zero/Const_9*
_output_shapes
: *
T0	
В
3linear/zero_fraction/total_zero/zero_count_9/SwitchSwitch'linear/zero_fraction/total_zero/Equal_9'linear/zero_fraction/total_zero/Equal_9*
T0
*
_output_shapes
: : 

5linear/zero_fraction/total_zero/zero_count_9/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_9/Switch:1*
_output_shapes
: *
T0


5linear/zero_fraction/total_zero/zero_count_9/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_9/Switch*
T0
*
_output_shapes
: 

4linear/zero_fraction/total_zero/zero_count_9/pred_idIdentity'linear/zero_fraction/total_zero/Equal_9*
T0
*
_output_shapes
: 
Џ
2linear/zero_fraction/total_zero/zero_count_9/ConstConst6^linear/zero_fraction/total_zero/zero_count_9/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0
К
?linear/zero_fraction/total_zero/zero_count_9/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_9/switch_f*
value
B	 Rр*
dtype0	*
_output_shapes
: 
Ф
Flinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_9/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ћ
Dlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_9/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
џ
Flinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
П
Hlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
Н
Hlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
К
Glinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
у
Slinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
У
Vlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqualNotEqualalinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1Slinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

: *
T0
ю
]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchFlinear/linear_model/feature_18_embedding/embedding_weights/part_0/read4linear/zero_fraction/total_zero/zero_count_9/pred_id*
T0*T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*(
_output_shapes
: : 

_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch_1Switch]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/SwitchGlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id*T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*(
_output_shapes
: : *
T0
ъ
Rlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

: *

DstT0
я
Slinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Ќ
[linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
й
Dlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
х
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Х
Xlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

: *
T0

_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/SwitchGlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id*
T0*T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*(
_output_shapes
: : 
ю
Tlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

: *

DstT0	
ё
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
В
]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 

Elinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	

Qlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_9/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
н
Rlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Э
Tlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_9/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
Ћ
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Ч
Clinear/zero_fraction/total_zero/zero_count_9/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Љ
4linear/zero_fraction/total_zero/zero_count_9/ToFloatCast;linear/zero_fraction/total_zero/zero_count_9/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	

;linear/zero_fraction/total_zero/zero_count_9/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_94linear/zero_fraction/total_zero/zero_count_9/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_9*
_output_shapes
: : 
г
0linear/zero_fraction/total_zero/zero_count_9/mulMulClinear/zero_fraction/total_zero/zero_count_9/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_9/ToFloat*
T0*
_output_shapes
: 
Э
2linear/zero_fraction/total_zero/zero_count_9/MergeMerge0linear/zero_fraction/total_zero/zero_count_9/mul2linear/zero_fraction/total_zero/zero_count_9/Const*
N*
_output_shapes
: : *
T0
j
(linear/zero_fraction/total_zero/Const_10Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ѕ
(linear/zero_fraction/total_zero/Equal_10Equal'linear/zero_fraction/total_size/Size_10(linear/zero_fraction/total_zero/Const_10*
T0	*
_output_shapes
: 
Е
4linear/zero_fraction/total_zero/zero_count_10/SwitchSwitch(linear/zero_fraction/total_zero/Equal_10(linear/zero_fraction/total_zero/Equal_10*
_output_shapes
: : *
T0


6linear/zero_fraction/total_zero/zero_count_10/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_10/Switch:1*
_output_shapes
: *
T0


6linear/zero_fraction/total_zero/zero_count_10/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_10/Switch*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_10/pred_idIdentity(linear/zero_fraction/total_zero/Equal_10*
_output_shapes
: *
T0

Б
3linear/zero_fraction/total_zero/zero_count_10/ConstConst7^linear/zero_fraction/total_zero/zero_count_10/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0
м
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOpReadVariableOpQlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
К
Qlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp/SwitchSwitch7linear/linear_model/feature_18_embedding/weights/part_05linear/zero_fraction/total_zero/zero_count_10/pred_id*
T0*J
_class@
><loc:@linear/linear_model/feature_18_embedding/weights/part_0*
_output_shapes
: : 
Л
@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_10/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_10/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ў
Elinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	

Glinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
С
Ilinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

П
Ilinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
М
Hlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
х
Tlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
Wlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Tlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:

^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp*(
_output_shapes
::
ь
Slinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
ё
Tlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
Џ
\linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
л
Elinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
ч
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
Ш
Ylinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:

`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp*(
_output_shapes
::
№
Ulinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
ѓ
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Е
^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ђ
Flinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Rlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
п
Slinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Я
Ulinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
Ў
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
Щ
Dlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Ћ
5linear/zero_fraction/total_zero/zero_count_10/ToFloatCast<linear/zero_fraction/total_zero/zero_count_10/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

<linear/zero_fraction/total_zero/zero_count_10/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_105linear/zero_fraction/total_zero/zero_count_10/pred_id*
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_10*
_output_shapes
: : 
ж
1linear/zero_fraction/total_zero/zero_count_10/mulMulDlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_10/ToFloat*
_output_shapes
: *
T0
а
3linear/zero_fraction/total_zero/zero_count_10/MergeMerge1linear/zero_fraction/total_zero/zero_count_10/mul3linear/zero_fraction/total_zero/zero_count_10/Const*
T0*
N*
_output_shapes
: : 
j
(linear/zero_fraction/total_zero/Const_11Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ѕ
(linear/zero_fraction/total_zero/Equal_11Equal'linear/zero_fraction/total_size/Size_11(linear/zero_fraction/total_zero/Const_11*
T0	*
_output_shapes
: 
Е
4linear/zero_fraction/total_zero/zero_count_11/SwitchSwitch(linear/zero_fraction/total_zero/Equal_11(linear/zero_fraction/total_zero/Equal_11*
_output_shapes
: : *
T0


6linear/zero_fraction/total_zero/zero_count_11/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_11/Switch:1*
T0
*
_output_shapes
: 

6linear/zero_fraction/total_zero/zero_count_11/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_11/Switch*
_output_shapes
: *
T0


5linear/zero_fraction/total_zero/zero_count_11/pred_idIdentity(linear/zero_fraction/total_zero/Equal_11*
T0
*
_output_shapes
: 
Б
3linear/zero_fraction/total_zero/zero_count_11/ConstConst7^linear/zero_fraction/total_zero/zero_count_11/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
М
@linear/zero_fraction/total_zero/zero_count_11/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_11/switch_f*
value
B	 R*
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_11/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_11/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ў
Elinear/zero_fraction/total_zero/zero_count_11/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_11/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	

Glinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_11/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_11/zero_fraction/LessEqual*
_output_shapes
: : *
T0

С
Ilinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
П
Ilinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
М
Hlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_11/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
х
Tlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
Ч
Wlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqualNotEqualblinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1Tlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes
:	
№
^linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchElinear/linear_model/feature_2_embedding/embedding_weights/part_0/read5linear/zero_fraction/total_zero/zero_count_11/pred_id*
T0*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0**
_output_shapes
:	:	

`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch_1Switch^linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/SwitchHlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_id*
T0*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0**
_output_shapes
:	:	
э
Slinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes
:	*

DstT0
ё
Tlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_t*
_output_shapes
:*
valueB"       *
dtype0
Џ
\linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
л
Elinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
ч
Vlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
Щ
Ylinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes
:	

`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch^linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/SwitchHlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_id*
T0*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0**
_output_shapes
:	:	
ё
Ulinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes
:	*

DstT0	
ѓ
Vlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Е
^linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ђ
Flinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	

Rlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_11/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
п
Slinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Я
Ulinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_11/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ў
Vlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Щ
Dlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Ћ
5linear/zero_fraction/total_zero/zero_count_11/ToFloatCast<linear/zero_fraction/total_zero/zero_count_11/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	

<linear/zero_fraction/total_zero/zero_count_11/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_115linear/zero_fraction/total_zero/zero_count_11/pred_id*
_output_shapes
: : *
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_11
ж
1linear/zero_fraction/total_zero/zero_count_11/mulMulDlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_11/ToFloat*
_output_shapes
: *
T0
а
3linear/zero_fraction/total_zero/zero_count_11/MergeMerge1linear/zero_fraction/total_zero/zero_count_11/mul3linear/zero_fraction/total_zero/zero_count_11/Const*
T0*
N*
_output_shapes
: : 
j
(linear/zero_fraction/total_zero/Const_12Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ѕ
(linear/zero_fraction/total_zero/Equal_12Equal'linear/zero_fraction/total_size/Size_12(linear/zero_fraction/total_zero/Const_12*
_output_shapes
: *
T0	
Е
4linear/zero_fraction/total_zero/zero_count_12/SwitchSwitch(linear/zero_fraction/total_zero/Equal_12(linear/zero_fraction/total_zero/Equal_12*
T0
*
_output_shapes
: : 

6linear/zero_fraction/total_zero/zero_count_12/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_12/Switch:1*
_output_shapes
: *
T0


6linear/zero_fraction/total_zero/zero_count_12/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_12/Switch*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_12/pred_idIdentity(linear/zero_fraction/total_zero/Equal_12*
_output_shapes
: *
T0

Б
3linear/zero_fraction/total_zero/zero_count_12/ConstConst7^linear/zero_fraction/total_zero/zero_count_12/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
м
Jlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOpReadVariableOpQlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
И
Qlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOp/SwitchSwitch6linear/linear_model/feature_2_embedding/weights/part_05linear/zero_fraction/total_zero/zero_count_12/pred_id*
T0*I
_class?
=;loc:@linear/linear_model/feature_2_embedding/weights/part_0*
_output_shapes
: : 
Л
@linear/zero_fraction/total_zero/zero_count_12/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_12/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_12/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_12/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ў
Elinear/zero_fraction/total_zero/zero_count_12/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_12/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 

Glinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_12/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_12/zero_fraction/LessEqual*
_output_shapes
: : *
T0

С
Ilinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
П
Ilinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
М
Hlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_12/zero_fraction/LessEqual*
_output_shapes
: *
T0

х
Tlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
Wlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Tlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:

^linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOp*(
_output_shapes
::
ь
Slinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/NotEqual*
_output_shapes

:*

DstT0*

SrcT0

ё
Tlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
Џ
\linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
л
Elinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
ч
Vlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш
Ylinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

:*
T0

`linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOp*(
_output_shapes
::
№
Ulinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/NotEqual*
_output_shapes

:*

DstT0	*

SrcT0

ѓ
Vlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Е
^linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ђ
Flinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	

Rlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_12/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
п
Slinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
Я
Ulinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_12/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ў
Vlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Щ
Dlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Ћ
5linear/zero_fraction/total_zero/zero_count_12/ToFloatCast<linear/zero_fraction/total_zero/zero_count_12/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

<linear/zero_fraction/total_zero/zero_count_12/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_125linear/zero_fraction/total_zero/zero_count_12/pred_id*
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_12*
_output_shapes
: : 
ж
1linear/zero_fraction/total_zero/zero_count_12/mulMulDlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_12/ToFloat*
T0*
_output_shapes
: 
а
3linear/zero_fraction/total_zero/zero_count_12/MergeMerge1linear/zero_fraction/total_zero/zero_count_12/mul3linear/zero_fraction/total_zero/zero_count_12/Const*
_output_shapes
: : *
T0*
N
j
(linear/zero_fraction/total_zero/Const_13Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ѕ
(linear/zero_fraction/total_zero/Equal_13Equal'linear/zero_fraction/total_size/Size_13(linear/zero_fraction/total_zero/Const_13*
T0	*
_output_shapes
: 
Е
4linear/zero_fraction/total_zero/zero_count_13/SwitchSwitch(linear/zero_fraction/total_zero/Equal_13(linear/zero_fraction/total_zero/Equal_13*
T0
*
_output_shapes
: : 

6linear/zero_fraction/total_zero/zero_count_13/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_13/Switch:1*
_output_shapes
: *
T0


6linear/zero_fraction/total_zero/zero_count_13/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_13/Switch*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_13/pred_idIdentity(linear/zero_fraction/total_zero/Equal_13*
T0
*
_output_shapes
: 
Б
3linear/zero_fraction/total_zero/zero_count_13/ConstConst7^linear/zero_fraction/total_zero/zero_count_13/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
м
Jlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOpReadVariableOpQlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
Є
Qlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOp/SwitchSwitch,linear/linear_model/feature_3/weights/part_05linear/zero_fraction/total_zero/zero_count_13/pred_id*
T0*?
_class5
31loc:@linear/linear_model/feature_3/weights/part_0*
_output_shapes
: : 
Л
@linear/zero_fraction/total_zero/zero_count_13/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_13/switch_f*
dtype0	*
_output_shapes
: *
value	B	 R
Ц
Glinear/zero_fraction/total_zero/zero_count_13/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_13/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ў
Elinear/zero_fraction/total_zero/zero_count_13/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_13/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	

Glinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_13/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_13/zero_fraction/LessEqual*
_output_shapes
: : *
T0

С
Ilinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

П
Ilinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
М
Hlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_13/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
х
Tlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
Ф
Wlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Tlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

:*
T0

^linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOp
ь
Slinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/NotEqual*
_output_shapes

:*

DstT0*

SrcT0

ё
Tlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Џ
\linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
л
Elinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
ч
Vlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш
Ylinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:

`linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOp*(
_output_shapes
::
№
Ulinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/NotEqual*
_output_shapes

:*

DstT0	*

SrcT0

ѓ
Vlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"       
Е
^linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
Ђ
Flinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Rlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_13/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
п
Slinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
Я
Ulinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_13/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ў
Vlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Щ
Dlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Ћ
5linear/zero_fraction/total_zero/zero_count_13/ToFloatCast<linear/zero_fraction/total_zero/zero_count_13/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	

<linear/zero_fraction/total_zero/zero_count_13/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_135linear/zero_fraction/total_zero/zero_count_13/pred_id*
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_13*
_output_shapes
: : 
ж
1linear/zero_fraction/total_zero/zero_count_13/mulMulDlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_13/ToFloat*
T0*
_output_shapes
: 
а
3linear/zero_fraction/total_zero/zero_count_13/MergeMerge1linear/zero_fraction/total_zero/zero_count_13/mul3linear/zero_fraction/total_zero/zero_count_13/Const*
T0*
N*
_output_shapes
: : 
j
(linear/zero_fraction/total_zero/Const_14Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ѕ
(linear/zero_fraction/total_zero/Equal_14Equal'linear/zero_fraction/total_size/Size_14(linear/zero_fraction/total_zero/Const_14*
_output_shapes
: *
T0	
Е
4linear/zero_fraction/total_zero/zero_count_14/SwitchSwitch(linear/zero_fraction/total_zero/Equal_14(linear/zero_fraction/total_zero/Equal_14*
T0
*
_output_shapes
: : 

6linear/zero_fraction/total_zero/zero_count_14/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_14/Switch:1*
T0
*
_output_shapes
: 

6linear/zero_fraction/total_zero/zero_count_14/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_14/Switch*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_14/pred_idIdentity(linear/zero_fraction/total_zero/Equal_14*
T0
*
_output_shapes
: 
Б
3linear/zero_fraction/total_zero/zero_count_14/ConstConst7^linear/zero_fraction/total_zero/zero_count_14/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
м
Jlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOpReadVariableOpQlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
Є
Qlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOp/SwitchSwitch,linear/linear_model/feature_4/weights/part_05linear/zero_fraction/total_zero/zero_count_14/pred_id*
_output_shapes
: : *
T0*?
_class5
31loc:@linear/linear_model/feature_4/weights/part_0
Л
@linear/zero_fraction/total_zero/zero_count_14/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_14/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_14/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_14/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ў
Elinear/zero_fraction/total_zero/zero_count_14/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_14/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	

Glinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_14/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_14/zero_fraction/LessEqual*
_output_shapes
: : *
T0

С
Ilinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

П
Ilinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/Switch*
_output_shapes
: *
T0

М
Hlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_14/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
х
Tlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
Wlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Tlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

:*
T0

^linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOp*(
_output_shapes
::
ь
Slinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
ё
Tlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Џ
\linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
л
Elinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
ч
Vlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Ш
Ylinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:

`linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOp*(
_output_shapes
::
№
Ulinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
ѓ
Vlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"       
Е
^linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ђ
Flinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Rlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_14/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
п
Slinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Я
Ulinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_14/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ў
Vlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
Щ
Dlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Ћ
5linear/zero_fraction/total_zero/zero_count_14/ToFloatCast<linear/zero_fraction/total_zero/zero_count_14/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	

<linear/zero_fraction/total_zero/zero_count_14/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_145linear/zero_fraction/total_zero/zero_count_14/pred_id*
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_14*
_output_shapes
: : 
ж
1linear/zero_fraction/total_zero/zero_count_14/mulMulDlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_14/ToFloat*
T0*
_output_shapes
: 
а
3linear/zero_fraction/total_zero/zero_count_14/MergeMerge1linear/zero_fraction/total_zero/zero_count_14/mul3linear/zero_fraction/total_zero/zero_count_14/Const*
T0*
N*
_output_shapes
: : 
j
(linear/zero_fraction/total_zero/Const_15Const*
_output_shapes
: *
value	B	 R *
dtype0	
Ѕ
(linear/zero_fraction/total_zero/Equal_15Equal'linear/zero_fraction/total_size/Size_15(linear/zero_fraction/total_zero/Const_15*
T0	*
_output_shapes
: 
Е
4linear/zero_fraction/total_zero/zero_count_15/SwitchSwitch(linear/zero_fraction/total_zero/Equal_15(linear/zero_fraction/total_zero/Equal_15*
_output_shapes
: : *
T0


6linear/zero_fraction/total_zero/zero_count_15/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_15/Switch:1*
_output_shapes
: *
T0


6linear/zero_fraction/total_zero/zero_count_15/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_15/Switch*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_15/pred_idIdentity(linear/zero_fraction/total_zero/Equal_15*
_output_shapes
: *
T0

Б
3linear/zero_fraction/total_zero/zero_count_15/ConstConst7^linear/zero_fraction/total_zero/zero_count_15/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0
м
Jlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOpReadVariableOpQlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
Є
Qlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOp/SwitchSwitch,linear/linear_model/feature_5/weights/part_05linear/zero_fraction/total_zero/zero_count_15/pred_id*
_output_shapes
: : *
T0*?
_class5
31loc:@linear/linear_model/feature_5/weights/part_0
Л
@linear/zero_fraction/total_zero/zero_count_15/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_15/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_15/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_15/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ў
Elinear/zero_fraction/total_zero/zero_count_15/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_15/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	

Glinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_15/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_15/zero_fraction/LessEqual*
_output_shapes
: : *
T0

С
Ilinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
П
Ilinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
М
Hlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_15/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
х
Tlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
Ф
Wlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Tlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:

^linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOp*(
_output_shapes
::
ь
Slinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
ё
Tlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Џ
\linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
л
Elinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
ч
Vlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш
Ylinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

:*
T0

`linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOp*(
_output_shapes
::
№
Ulinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
ѓ
Vlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Е
^linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ђ
Flinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Rlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_15/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
п
Slinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Я
Ulinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_15/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ў
Vlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Щ
Dlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Ћ
5linear/zero_fraction/total_zero/zero_count_15/ToFloatCast<linear/zero_fraction/total_zero/zero_count_15/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

<linear/zero_fraction/total_zero/zero_count_15/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_155linear/zero_fraction/total_zero/zero_count_15/pred_id*
_output_shapes
: : *
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_15
ж
1linear/zero_fraction/total_zero/zero_count_15/mulMulDlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_15/ToFloat*
T0*
_output_shapes
: 
а
3linear/zero_fraction/total_zero/zero_count_15/MergeMerge1linear/zero_fraction/total_zero/zero_count_15/mul3linear/zero_fraction/total_zero/zero_count_15/Const*
T0*
N*
_output_shapes
: : 
j
(linear/zero_fraction/total_zero/Const_16Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ѕ
(linear/zero_fraction/total_zero/Equal_16Equal'linear/zero_fraction/total_size/Size_16(linear/zero_fraction/total_zero/Const_16*
T0	*
_output_shapes
: 
Е
4linear/zero_fraction/total_zero/zero_count_16/SwitchSwitch(linear/zero_fraction/total_zero/Equal_16(linear/zero_fraction/total_zero/Equal_16*
_output_shapes
: : *
T0


6linear/zero_fraction/total_zero/zero_count_16/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_16/Switch:1*
_output_shapes
: *
T0


6linear/zero_fraction/total_zero/zero_count_16/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_16/Switch*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_16/pred_idIdentity(linear/zero_fraction/total_zero/Equal_16*
T0
*
_output_shapes
: 
Б
3linear/zero_fraction/total_zero/zero_count_16/ConstConst7^linear/zero_fraction/total_zero/zero_count_16/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
м
Jlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOpReadVariableOpQlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
Є
Qlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOp/SwitchSwitch,linear/linear_model/feature_6/weights/part_05linear/zero_fraction/total_zero/zero_count_16/pred_id*
_output_shapes
: : *
T0*?
_class5
31loc:@linear/linear_model/feature_6/weights/part_0
Л
@linear/zero_fraction/total_zero/zero_count_16/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_16/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_16/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_16/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ў
Elinear/zero_fraction/total_zero/zero_count_16/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_16/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	

Glinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_16/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_16/zero_fraction/LessEqual*
_output_shapes
: : *
T0

С
Ilinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
П
Ilinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/Switch*
_output_shapes
: *
T0

М
Hlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_16/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
х
Tlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
Wlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Tlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

:*
T0

^linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOp
ь
Slinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
ё
Tlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Џ
\linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
л
Elinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
ч
Vlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш
Ylinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:

`linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOp*(
_output_shapes
::
№
Ulinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/NotEqual*
_output_shapes

:*

DstT0	*

SrcT0

ѓ
Vlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Е
^linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ђ
Flinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	

Rlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_16/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
п
Slinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
Я
Ulinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_16/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ў
Vlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Щ
Dlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Ћ
5linear/zero_fraction/total_zero/zero_count_16/ToFloatCast<linear/zero_fraction/total_zero/zero_count_16/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

<linear/zero_fraction/total_zero/zero_count_16/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_165linear/zero_fraction/total_zero/zero_count_16/pred_id*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_16*
_output_shapes
: : *
T0	
ж
1linear/zero_fraction/total_zero/zero_count_16/mulMulDlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_16/ToFloat*
_output_shapes
: *
T0
а
3linear/zero_fraction/total_zero/zero_count_16/MergeMerge1linear/zero_fraction/total_zero/zero_count_16/mul3linear/zero_fraction/total_zero/zero_count_16/Const*
N*
_output_shapes
: : *
T0
j
(linear/zero_fraction/total_zero/Const_17Const*
_output_shapes
: *
value	B	 R *
dtype0	
Ѕ
(linear/zero_fraction/total_zero/Equal_17Equal'linear/zero_fraction/total_size/Size_17(linear/zero_fraction/total_zero/Const_17*
_output_shapes
: *
T0	
Е
4linear/zero_fraction/total_zero/zero_count_17/SwitchSwitch(linear/zero_fraction/total_zero/Equal_17(linear/zero_fraction/total_zero/Equal_17*
T0
*
_output_shapes
: : 

6linear/zero_fraction/total_zero/zero_count_17/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_17/Switch:1*
T0
*
_output_shapes
: 

6linear/zero_fraction/total_zero/zero_count_17/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_17/Switch*
_output_shapes
: *
T0


5linear/zero_fraction/total_zero/zero_count_17/pred_idIdentity(linear/zero_fraction/total_zero/Equal_17*
T0
*
_output_shapes
: 
Б
3linear/zero_fraction/total_zero/zero_count_17/ConstConst7^linear/zero_fraction/total_zero/zero_count_17/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Л
@linear/zero_fraction/total_zero/zero_count_17/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_17/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_17/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_17/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ў
Elinear/zero_fraction/total_zero/zero_count_17/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_17/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 

Glinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_17/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_17/zero_fraction/LessEqual*
_output_shapes
: : *
T0

С
Ilinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
П
Ilinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
М
Hlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_17/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
х
Tlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
Wlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqualNotEqualblinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1Tlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

:*
T0
ю
^linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchElinear/linear_model/feature_7_embedding/embedding_weights/part_0/read5linear/zero_fraction/total_zero/zero_count_17/pred_id*
T0*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0*(
_output_shapes
::

`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch_1Switch^linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/SwitchHlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0
ь
Slinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
ё
Tlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_t*
_output_shapes
:*
valueB"       *
dtype0
Џ
\linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
л
Elinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
ч
Vlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш
Ylinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

:*
T0

`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch^linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/SwitchHlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_id*
T0*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0*(
_output_shapes
::
№
Ulinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/NotEqual*
_output_shapes

:*

DstT0	*

SrcT0

ѓ
Vlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Е
^linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ђ
Flinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/Cast*
_output_shapes
: : *
T0	*
N

Rlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_17/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
п
Slinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
Я
Ulinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_17/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ў
Vlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Щ
Dlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Ћ
5linear/zero_fraction/total_zero/zero_count_17/ToFloatCast<linear/zero_fraction/total_zero/zero_count_17/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	

<linear/zero_fraction/total_zero/zero_count_17/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_175linear/zero_fraction/total_zero/zero_count_17/pred_id*
_output_shapes
: : *
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_17
ж
1linear/zero_fraction/total_zero/zero_count_17/mulMulDlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_17/ToFloat*
T0*
_output_shapes
: 
а
3linear/zero_fraction/total_zero/zero_count_17/MergeMerge1linear/zero_fraction/total_zero/zero_count_17/mul3linear/zero_fraction/total_zero/zero_count_17/Const*
N*
_output_shapes
: : *
T0
j
(linear/zero_fraction/total_zero/Const_18Const*
dtype0	*
_output_shapes
: *
value	B	 R 
Ѕ
(linear/zero_fraction/total_zero/Equal_18Equal'linear/zero_fraction/total_size/Size_18(linear/zero_fraction/total_zero/Const_18*
T0	*
_output_shapes
: 
Е
4linear/zero_fraction/total_zero/zero_count_18/SwitchSwitch(linear/zero_fraction/total_zero/Equal_18(linear/zero_fraction/total_zero/Equal_18*
_output_shapes
: : *
T0


6linear/zero_fraction/total_zero/zero_count_18/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_18/Switch:1*
T0
*
_output_shapes
: 

6linear/zero_fraction/total_zero/zero_count_18/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_18/Switch*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_18/pred_idIdentity(linear/zero_fraction/total_zero/Equal_18*
_output_shapes
: *
T0

Б
3linear/zero_fraction/total_zero/zero_count_18/ConstConst7^linear/zero_fraction/total_zero/zero_count_18/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
м
Jlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOpReadVariableOpQlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
И
Qlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOp/SwitchSwitch6linear/linear_model/feature_7_embedding/weights/part_05linear/zero_fraction/total_zero/zero_count_18/pred_id*
T0*I
_class?
=;loc:@linear/linear_model/feature_7_embedding/weights/part_0*
_output_shapes
: : 
Л
@linear/zero_fraction/total_zero/zero_count_18/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_18/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_18/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_18/switch_f*
_output_shapes
: *
valueB	 Rџџџџ*
dtype0	
ў
Elinear/zero_fraction/total_zero/zero_count_18/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_18/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 

Glinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_18/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_18/zero_fraction/LessEqual*
_output_shapes
: : *
T0

С
Ilinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

П
Ilinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
М
Hlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_18/zero_fraction/LessEqual*
_output_shapes
: *
T0

х
Tlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
Wlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Tlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:

^linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOp
ь
Slinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
ё
Tlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Џ
\linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
л
Elinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
ч
Vlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш
Ylinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

:*
T0

`linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOp
№
Ulinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
ѓ
Vlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Е
^linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ђ
Flinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Rlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_18/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
п
Slinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Я
Ulinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_18/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
Ў
Vlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Щ
Dlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Ћ
5linear/zero_fraction/total_zero/zero_count_18/ToFloatCast<linear/zero_fraction/total_zero/zero_count_18/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	

<linear/zero_fraction/total_zero/zero_count_18/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_185linear/zero_fraction/total_zero/zero_count_18/pred_id*
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_18*
_output_shapes
: : 
ж
1linear/zero_fraction/total_zero/zero_count_18/mulMulDlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_18/ToFloat*
T0*
_output_shapes
: 
а
3linear/zero_fraction/total_zero/zero_count_18/MergeMerge1linear/zero_fraction/total_zero/zero_count_18/mul3linear/zero_fraction/total_zero/zero_count_18/Const*
_output_shapes
: : *
T0*
N
j
(linear/zero_fraction/total_zero/Const_19Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ѕ
(linear/zero_fraction/total_zero/Equal_19Equal'linear/zero_fraction/total_size/Size_19(linear/zero_fraction/total_zero/Const_19*
_output_shapes
: *
T0	
Е
4linear/zero_fraction/total_zero/zero_count_19/SwitchSwitch(linear/zero_fraction/total_zero/Equal_19(linear/zero_fraction/total_zero/Equal_19*
T0
*
_output_shapes
: : 

6linear/zero_fraction/total_zero/zero_count_19/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_19/Switch:1*
_output_shapes
: *
T0


6linear/zero_fraction/total_zero/zero_count_19/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_19/Switch*
_output_shapes
: *
T0


5linear/zero_fraction/total_zero/zero_count_19/pred_idIdentity(linear/zero_fraction/total_zero/Equal_19*
_output_shapes
: *
T0

Б
3linear/zero_fraction/total_zero/zero_count_19/ConstConst7^linear/zero_fraction/total_zero/zero_count_19/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
М
@linear/zero_fraction/total_zero/zero_count_19/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_19/switch_f*
value
B	 RЇ*
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_19/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_19/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ў
Elinear/zero_fraction/total_zero/zero_count_19/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_19/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	

Glinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_19/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_19/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
С
Ilinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
П
Ilinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
М
Hlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_19/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
х
Tlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
Wlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqualNotEqualblinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1Tlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
ю
^linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchElinear/linear_model/feature_8_embedding/embedding_weights/part_0/read5linear/zero_fraction/total_zero/zero_count_19/pred_id*
T0*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*(
_output_shapes
::

`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch_1Switch^linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/SwitchHlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_id*
T0*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*(
_output_shapes
::
ь
Slinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
ё
Tlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Џ
\linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
л
Elinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
ч
Vlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Ш
Ylinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:

`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch^linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/SwitchHlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_id*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*(
_output_shapes
::*
T0
№
Ulinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
ѓ
Vlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_f*
_output_shapes
:*
valueB"       *
dtype0
Е
^linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
Ђ
Flinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Rlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_19/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
п
Slinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Я
Ulinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_19/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ў
Vlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
Щ
Dlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Ћ
5linear/zero_fraction/total_zero/zero_count_19/ToFloatCast<linear/zero_fraction/total_zero/zero_count_19/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	

<linear/zero_fraction/total_zero/zero_count_19/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_195linear/zero_fraction/total_zero/zero_count_19/pred_id*
_output_shapes
: : *
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_19
ж
1linear/zero_fraction/total_zero/zero_count_19/mulMulDlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_19/ToFloat*
T0*
_output_shapes
: 
а
3linear/zero_fraction/total_zero/zero_count_19/MergeMerge1linear/zero_fraction/total_zero/zero_count_19/mul3linear/zero_fraction/total_zero/zero_count_19/Const*
T0*
N*
_output_shapes
: : 
j
(linear/zero_fraction/total_zero/Const_20Const*
dtype0	*
_output_shapes
: *
value	B	 R 
Ѕ
(linear/zero_fraction/total_zero/Equal_20Equal'linear/zero_fraction/total_size/Size_20(linear/zero_fraction/total_zero/Const_20*
T0	*
_output_shapes
: 
Е
4linear/zero_fraction/total_zero/zero_count_20/SwitchSwitch(linear/zero_fraction/total_zero/Equal_20(linear/zero_fraction/total_zero/Equal_20*
T0
*
_output_shapes
: : 

6linear/zero_fraction/total_zero/zero_count_20/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_20/Switch:1*
T0
*
_output_shapes
: 

6linear/zero_fraction/total_zero/zero_count_20/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_20/Switch*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_20/pred_idIdentity(linear/zero_fraction/total_zero/Equal_20*
_output_shapes
: *
T0

Б
3linear/zero_fraction/total_zero/zero_count_20/ConstConst7^linear/zero_fraction/total_zero/zero_count_20/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
м
Jlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOpReadVariableOpQlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
И
Qlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOp/SwitchSwitch6linear/linear_model/feature_8_embedding/weights/part_05linear/zero_fraction/total_zero/zero_count_20/pred_id*
T0*I
_class?
=;loc:@linear/linear_model/feature_8_embedding/weights/part_0*
_output_shapes
: : 
Л
@linear/zero_fraction/total_zero/zero_count_20/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_20/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_20/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_20/switch_f*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 
ў
Elinear/zero_fraction/total_zero/zero_count_20/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_20/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 

Glinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_20/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_20/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
С
Ilinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

П
Ilinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
М
Hlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_20/zero_fraction/LessEqual*
_output_shapes
: *
T0

х
Tlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
Wlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Tlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

:*
T0

^linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOp
ь
Slinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
ё
Tlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Џ
\linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
л
Elinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
ч
Vlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш
Ylinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:

`linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOp
№
Ulinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/NotEqual*
_output_shapes

:*

DstT0	*

SrcT0

ѓ
Vlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Е
^linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ђ
Flinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/Cast*
_output_shapes
: : *
T0	*
N

Rlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_20/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
п
Slinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
Я
Ulinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_20/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
Ў
Vlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
Щ
Dlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
Ћ
5linear/zero_fraction/total_zero/zero_count_20/ToFloatCast<linear/zero_fraction/total_zero/zero_count_20/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

<linear/zero_fraction/total_zero/zero_count_20/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_205linear/zero_fraction/total_zero/zero_count_20/pred_id*
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_20*
_output_shapes
: : 
ж
1linear/zero_fraction/total_zero/zero_count_20/mulMulDlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_20/ToFloat*
T0*
_output_shapes
: 
а
3linear/zero_fraction/total_zero/zero_count_20/MergeMerge1linear/zero_fraction/total_zero/zero_count_20/mul3linear/zero_fraction/total_zero/zero_count_20/Const*
_output_shapes
: : *
T0*
N
j
(linear/zero_fraction/total_zero/Const_21Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ѕ
(linear/zero_fraction/total_zero/Equal_21Equal'linear/zero_fraction/total_size/Size_21(linear/zero_fraction/total_zero/Const_21*
T0	*
_output_shapes
: 
Е
4linear/zero_fraction/total_zero/zero_count_21/SwitchSwitch(linear/zero_fraction/total_zero/Equal_21(linear/zero_fraction/total_zero/Equal_21*
_output_shapes
: : *
T0


6linear/zero_fraction/total_zero/zero_count_21/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_21/Switch:1*
T0
*
_output_shapes
: 

6linear/zero_fraction/total_zero/zero_count_21/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_21/Switch*
T0
*
_output_shapes
: 

5linear/zero_fraction/total_zero/zero_count_21/pred_idIdentity(linear/zero_fraction/total_zero/Equal_21*
_output_shapes
: *
T0

Б
3linear/zero_fraction/total_zero/zero_count_21/ConstConst7^linear/zero_fraction/total_zero/zero_count_21/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
М
@linear/zero_fraction/total_zero/zero_count_21/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_21/switch_f*
value
B	 R *
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_21/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_21/switch_f*
dtype0	*
_output_shapes
: *
valueB	 Rџџџџ
ў
Elinear/zero_fraction/total_zero/zero_count_21/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_21/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	

Glinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_21/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_21/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
С
Ilinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
П
Ilinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/Switch*
_output_shapes
: *
T0

М
Hlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_21/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
х
Tlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Ч
Wlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqualNotEqualblinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1Tlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes
:	Ѕ
№
^linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchElinear/linear_model/feature_9_embedding/embedding_weights/part_0/read5linear/zero_fraction/total_zero/zero_count_21/pred_id*
T0*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0**
_output_shapes
:	Ѕ:	Ѕ

`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch_1Switch^linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/SwitchHlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_id**
_output_shapes
:	Ѕ:	Ѕ*
T0*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0
э
Slinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual*
_output_shapes
:	Ѕ*

DstT0*

SrcT0

ё
Tlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Џ
\linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
л
Elinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
ч
Vlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
Щ
Ylinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes
:	Ѕ*
T0

`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitch^linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/SwitchHlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_id**
_output_shapes
:	Ѕ:	Ѕ*
T0*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0
ё
Ulinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes
:	Ѕ*

DstT0	
ѓ
Vlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Е
^linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
Ђ
Flinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Rlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_21/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
п
Slinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Я
Ulinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_21/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ў
Vlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
Щ
Dlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Ћ
5linear/zero_fraction/total_zero/zero_count_21/ToFloatCast<linear/zero_fraction/total_zero/zero_count_21/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

<linear/zero_fraction/total_zero/zero_count_21/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_215linear/zero_fraction/total_zero/zero_count_21/pred_id*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_21*
_output_shapes
: : *
T0	
ж
1linear/zero_fraction/total_zero/zero_count_21/mulMulDlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_21/ToFloat*
T0*
_output_shapes
: 
а
3linear/zero_fraction/total_zero/zero_count_21/MergeMerge1linear/zero_fraction/total_zero/zero_count_21/mul3linear/zero_fraction/total_zero/zero_count_21/Const*
N*
_output_shapes
: : *
T0
j
(linear/zero_fraction/total_zero/Const_22Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Ѕ
(linear/zero_fraction/total_zero/Equal_22Equal'linear/zero_fraction/total_size/Size_22(linear/zero_fraction/total_zero/Const_22*
_output_shapes
: *
T0	
Е
4linear/zero_fraction/total_zero/zero_count_22/SwitchSwitch(linear/zero_fraction/total_zero/Equal_22(linear/zero_fraction/total_zero/Equal_22*
T0
*
_output_shapes
: : 

6linear/zero_fraction/total_zero/zero_count_22/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_22/Switch:1*
T0
*
_output_shapes
: 

6linear/zero_fraction/total_zero/zero_count_22/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_22/Switch*
_output_shapes
: *
T0


5linear/zero_fraction/total_zero/zero_count_22/pred_idIdentity(linear/zero_fraction/total_zero/Equal_22*
_output_shapes
: *
T0

Б
3linear/zero_fraction/total_zero/zero_count_22/ConstConst7^linear/zero_fraction/total_zero/zero_count_22/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
м
Jlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOpReadVariableOpQlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
И
Qlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOp/SwitchSwitch6linear/linear_model/feature_9_embedding/weights/part_05linear/zero_fraction/total_zero/zero_count_22/pred_id*
T0*I
_class?
=;loc:@linear/linear_model/feature_9_embedding/weights/part_0*
_output_shapes
: : 
Л
@linear/zero_fraction/total_zero/zero_count_22/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_22/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
Ц
Glinear/zero_fraction/total_zero/zero_count_22/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_22/switch_f*
_output_shapes
: *
valueB	 Rџџџџ*
dtype0	
ў
Elinear/zero_fraction/total_zero/zero_count_22/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_22/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 

Glinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_22/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_22/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
С
Ilinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

П
Ilinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/Switch*
_output_shapes
: *
T0

М
Hlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_22/zero_fraction/LessEqual*
_output_shapes
: *
T0

х
Tlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
Wlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Tlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

:*
T0

^linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOp*(
_output_shapes
::
ь
Slinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
ё
Tlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_t*
_output_shapes
:*
valueB"       *
dtype0
Џ
\linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
л
Elinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
ч
Vlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Ш
Ylinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

:*
T0

`linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOp
№
Ulinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/NotEqual*
_output_shapes

:*

DstT0	*

SrcT0

ѓ
Vlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Е
^linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ђ
Flinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

Rlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_22/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
п
Slinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
Я
Ulinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_22/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
Ў
Vlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
Щ
Dlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
Ћ
5linear/zero_fraction/total_zero/zero_count_22/ToFloatCast<linear/zero_fraction/total_zero/zero_count_22/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0

<linear/zero_fraction/total_zero/zero_count_22/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_225linear/zero_fraction/total_zero/zero_count_22/pred_id*
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_22*
_output_shapes
: : 
ж
1linear/zero_fraction/total_zero/zero_count_22/mulMulDlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_22/ToFloat*
T0*
_output_shapes
: 
а
3linear/zero_fraction/total_zero/zero_count_22/MergeMerge1linear/zero_fraction/total_zero/zero_count_22/mul3linear/zero_fraction/total_zero/zero_count_22/Const*
T0*
N*
_output_shapes
: : 


$linear/zero_fraction/total_zero/AddNAddN0linear/zero_fraction/total_zero/zero_count/Merge2linear/zero_fraction/total_zero/zero_count_1/Merge2linear/zero_fraction/total_zero/zero_count_2/Merge2linear/zero_fraction/total_zero/zero_count_3/Merge2linear/zero_fraction/total_zero/zero_count_4/Merge2linear/zero_fraction/total_zero/zero_count_5/Merge2linear/zero_fraction/total_zero/zero_count_6/Merge2linear/zero_fraction/total_zero/zero_count_7/Merge2linear/zero_fraction/total_zero/zero_count_8/Merge2linear/zero_fraction/total_zero/zero_count_9/Merge3linear/zero_fraction/total_zero/zero_count_10/Merge3linear/zero_fraction/total_zero/zero_count_11/Merge3linear/zero_fraction/total_zero/zero_count_12/Merge3linear/zero_fraction/total_zero/zero_count_13/Merge3linear/zero_fraction/total_zero/zero_count_14/Merge3linear/zero_fraction/total_zero/zero_count_15/Merge3linear/zero_fraction/total_zero/zero_count_16/Merge3linear/zero_fraction/total_zero/zero_count_17/Merge3linear/zero_fraction/total_zero/zero_count_18/Merge3linear/zero_fraction/total_zero/zero_count_19/Merge3linear/zero_fraction/total_zero/zero_count_20/Merge3linear/zero_fraction/total_zero/zero_count_21/Merge3linear/zero_fraction/total_zero/zero_count_22/Merge*
T0*
N*
_output_shapes
: 

)linear/zero_fraction/compute/float32_sizeCast$linear/zero_fraction/total_size/AddN*

SrcT0	*
_output_shapes
: *

DstT0
Ё
$linear/zero_fraction/compute/truedivRealDiv$linear/zero_fraction/total_zero/AddN)linear/zero_fraction/compute/float32_size*
_output_shapes
: *
T0
|
)linear/zero_fraction/zero_fraction_or_nanIdentity$linear/zero_fraction/compute/truediv*
T0*
_output_shapes
: 

$linear/fraction_of_zero_weights/tagsConst*0
value'B% Blinear/fraction_of_zero_weights*
dtype0*
_output_shapes
: 
Ђ
linear/fraction_of_zero_weightsScalarSummary$linear/fraction_of_zero_weights/tags)linear/zero_fraction/zero_fraction_or_nan*
_output_shapes
: *
T0

linear/zero_fraction_1/SizeSize:linear/linear_model/linear_model/linear_model/weighted_sum*
out_type0	*
_output_shapes
: *
T0
h
"linear/zero_fraction_1/LessEqual/yConst*
valueB	 Rџџџџ*
dtype0	*
_output_shapes
: 

 linear/zero_fraction_1/LessEqual	LessEquallinear/zero_fraction_1/Size"linear/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 

"linear/zero_fraction_1/cond/SwitchSwitch linear/zero_fraction_1/LessEqual linear/zero_fraction_1/LessEqual*
_output_shapes
: : *
T0

w
$linear/zero_fraction_1/cond/switch_tIdentity$linear/zero_fraction_1/cond/Switch:1*
_output_shapes
: *
T0

u
$linear/zero_fraction_1/cond/switch_fIdentity"linear/zero_fraction_1/cond/Switch*
T0
*
_output_shapes
: 
r
#linear/zero_fraction_1/cond/pred_idIdentity linear/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: 

/linear/zero_fraction_1/cond/count_nonzero/zerosConst%^linear/zero_fraction_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
о
2linear/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual;linear/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1/linear/zero_fraction_1/cond/count_nonzero/zeros*
T0*'
_output_shapes
:џџџџџџџџџ
И
9linear/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitch:linear/linear_model/linear_model/linear_model/weighted_sum#linear/zero_fraction_1/cond/pred_id*
T0*M
_classC
A?loc:@linear/linear_model/linear_model/linear_model/weighted_sum*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
Ћ
.linear/zero_fraction_1/cond/count_nonzero/CastCast2linear/zero_fraction_1/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:џџџџџџџџџ*

DstT0
Ї
/linear/zero_fraction_1/cond/count_nonzero/ConstConst%^linear/zero_fraction_1/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
Р
7linear/zero_fraction_1/cond/count_nonzero/nonzero_countSum.linear/zero_fraction_1/cond/count_nonzero/Cast/linear/zero_fraction_1/cond/count_nonzero/Const*
_output_shapes
: *
T0

 linear/zero_fraction_1/cond/CastCast7linear/zero_fraction_1/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	

1linear/zero_fraction_1/cond/count_nonzero_1/zerosConst%^linear/zero_fraction_1/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
т
4linear/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual;linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch1linear/zero_fraction_1/cond/count_nonzero_1/zeros*'
_output_shapes
:џџџџџџџџџ*
T0
К
;linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitch:linear/linear_model/linear_model/linear_model/weighted_sum#linear/zero_fraction_1/cond/pred_id*M
_classC
A?loc:@linear/linear_model/linear_model/linear_model/weighted_sum*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
T0
Џ
0linear/zero_fraction_1/cond/count_nonzero_1/CastCast4linear/zero_fraction_1/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:џџџџџџџџџ*

DstT0	
Љ
1linear/zero_fraction_1/cond/count_nonzero_1/ConstConst%^linear/zero_fraction_1/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
Ц
9linear/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum0linear/zero_fraction_1/cond/count_nonzero_1/Cast1linear/zero_fraction_1/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
Г
!linear/zero_fraction_1/cond/MergeMerge9linear/zero_fraction_1/cond/count_nonzero_1/nonzero_count linear/zero_fraction_1/cond/Cast*
T0	*
N*
_output_shapes
: : 

-linear/zero_fraction_1/counts_to_fraction/subSublinear/zero_fraction_1/Size!linear/zero_fraction_1/cond/Merge*
T0	*
_output_shapes
: 

.linear/zero_fraction_1/counts_to_fraction/CastCast-linear/zero_fraction_1/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	

0linear/zero_fraction_1/counts_to_fraction/Cast_1Castlinear/zero_fraction_1/Size*

SrcT0	*
_output_shapes
: *

DstT0
П
1linear/zero_fraction_1/counts_to_fraction/truedivRealDiv.linear/zero_fraction_1/counts_to_fraction/Cast0linear/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 

linear/zero_fraction_1/fractionIdentity1linear/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 

*linear/linear/fraction_of_zero_values/tagsConst*6
value-B+ B%linear/linear/fraction_of_zero_values*
dtype0*
_output_shapes
: 
Є
%linear/linear/fraction_of_zero_valuesScalarSummary*linear/linear/fraction_of_zero_values/tagslinear/zero_fraction_1/fraction*
_output_shapes
: *
T0
u
linear/linear/activation/tagConst*
_output_shapes
: *)
value B Blinear/linear/activation*
dtype0

linear/linear/activationHistogramSummarylinear/linear/activation/tag:linear/linear_model/linear_model/linear_model/weighted_sum*
_output_shapes
: 

addAdddnn/logits/BiasAdd:linear/linear_model/linear_model/linear_model/weighted_sum*'
_output_shapes
:џџџџџџџџџ*
T0
D
head/logits/ShapeShapeadd*
T0*
_output_shapes
:
g
%head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
W
Ohead/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
H
@head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp

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
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
~
save/Read/ReadVariableOpReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0*
dtype0*
_output_shapes
:
X
save/IdentityIdentitysave/Read/ReadVariableOp*
T0*
_output_shapes
:
^
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes
:

save/Read_1/ReadVariableOpReadVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0*
dtype0*
_output_shapes
:
\
save/Identity_2Identitysave/Read_1/ReadVariableOp*
_output_shapes
:*
T0
`
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
T0*
_output_shapes
:
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
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
T0*
_output_shapes
:
{
save/Read_3/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:	
a
save/Identity_6Identitysave/Read_3/ReadVariableOp*
T0*
_output_shapes
:	
e
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
_output_shapes
:	*
T0

save/Read_4/ReadVariableOpReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0*
dtype0*
_output_shapes
:
\
save/Identity_8Identitysave/Read_4/ReadVariableOp*
_output_shapes
:*
T0
`
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
T0*
_output_shapes
:

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
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
_output_shapes
:*
T0
z
save/Read_7/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:*
dtype0
a
save/Identity_14Identitysave/Read_7/ReadVariableOp*
_output_shapes

:*
T0
f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
T0*
_output_shapes

:
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
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
T0*
_output_shapes
:
s
save/Read_9/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:
a
save/Identity_18Identitysave/Read_9/ReadVariableOp*
_output_shapes

:*
T0
f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
T0*
_output_shapes

:

save/Read_10/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
^
save/Identity_20Identitysave/Read_10/ReadVariableOp*
_output_shapes
:*
T0
b
save/Identity_21Identitysave/Identity_20"/device:CPU:0*
T0*
_output_shapes
:

save/Read_11/ReadVariableOpReadVariableOp-linear/linear_model/feature_10/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_22Identitysave/Read_11/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_23Identitysave/Identity_22"/device:CPU:0*
T0*
_output_shapes

:

save/Read_12/ReadVariableOpReadVariableOp-linear/linear_model/feature_11/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_24Identitysave/Read_12/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_25Identitysave/Identity_24"/device:CPU:0*
_output_shapes

:*
T0

save/Read_13/ReadVariableOpReadVariableOp-linear/linear_model/feature_12/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_26Identitysave/Read_13/ReadVariableOp*
_output_shapes

:*
T0
f
save/Identity_27Identitysave/Identity_26"/device:CPU:0*
T0*
_output_shapes

:

save/Read_14/ReadVariableOpReadVariableOp-linear/linear_model/feature_13/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_28Identitysave/Read_14/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_29Identitysave/Identity_28"/device:CPU:0*
T0*
_output_shapes

:

save/Read_15/ReadVariableOpReadVariableOp7linear/linear_model/feature_14_embedding/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_30Identitysave/Read_15/ReadVariableOp*
_output_shapes

:*
T0
f
save/Identity_31Identitysave/Identity_30"/device:CPU:0*
T0*
_output_shapes

:

save/Read_16/ReadVariableOpReadVariableOp-linear/linear_model/feature_15/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_32Identitysave/Read_16/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_33Identitysave/Identity_32"/device:CPU:0*
_output_shapes

:*
T0

save/Read_17/ReadVariableOpReadVariableOp-linear/linear_model/feature_16/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_34Identitysave/Read_17/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_35Identitysave/Identity_34"/device:CPU:0*
T0*
_output_shapes

:

save/Read_18/ReadVariableOpReadVariableOp-linear/linear_model/feature_17/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_36Identitysave/Read_18/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_37Identitysave/Identity_36"/device:CPU:0*
_output_shapes

:*
T0

save/Read_19/ReadVariableOpReadVariableOp7linear/linear_model/feature_18_embedding/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_38Identitysave/Read_19/ReadVariableOp*
_output_shapes

:*
T0
f
save/Identity_39Identitysave/Identity_38"/device:CPU:0*
_output_shapes

:*
T0

save/Read_20/ReadVariableOpReadVariableOp6linear/linear_model/feature_2_embedding/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_40Identitysave/Read_20/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_41Identitysave/Identity_40"/device:CPU:0*
_output_shapes

:*
T0

save/Read_21/ReadVariableOpReadVariableOp,linear/linear_model/feature_3/weights/part_0*
_output_shapes

:*
dtype0
b
save/Identity_42Identitysave/Read_21/ReadVariableOp*
_output_shapes

:*
T0
f
save/Identity_43Identitysave/Identity_42"/device:CPU:0*
T0*
_output_shapes

:

save/Read_22/ReadVariableOpReadVariableOp,linear/linear_model/feature_4/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_44Identitysave/Read_22/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_45Identitysave/Identity_44"/device:CPU:0*
T0*
_output_shapes

:

save/Read_23/ReadVariableOpReadVariableOp,linear/linear_model/feature_5/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_46Identitysave/Read_23/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_47Identitysave/Identity_46"/device:CPU:0*
_output_shapes

:*
T0

save/Read_24/ReadVariableOpReadVariableOp,linear/linear_model/feature_6/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_48Identitysave/Read_24/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_49Identitysave/Identity_48"/device:CPU:0*
T0*
_output_shapes

:

save/Read_25/ReadVariableOpReadVariableOp6linear/linear_model/feature_7_embedding/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_50Identitysave/Read_25/ReadVariableOp*
_output_shapes

:*
T0
f
save/Identity_51Identitysave/Identity_50"/device:CPU:0*
T0*
_output_shapes

:

save/Read_26/ReadVariableOpReadVariableOp6linear/linear_model/feature_8_embedding/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_52Identitysave/Read_26/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_53Identitysave/Identity_52"/device:CPU:0*
T0*
_output_shapes

:

save/Read_27/ReadVariableOpReadVariableOp6linear/linear_model/feature_9_embedding/weights/part_0*
_output_shapes

:*
dtype0
b
save/Identity_54Identitysave/Read_27/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_55Identitysave/Identity_54"/device:CPU:0*
T0*
_output_shapes

:

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_582915e1ea9d47d9b3f2b3ef601389fd/part*
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
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
	
save/SaveV2/tensor_namesConst"/device:CPU:0*Ј
valueBB)dnn/hiddenlayer_0/batchnorm_0/moving_meanB-dnn/hiddenlayer_0/batchnorm_0/moving_varianceB)dnn/hiddenlayer_1/batchnorm_1/moving_meanB-dnn/hiddenlayer_1/batchnorm_1/moving_varianceBQdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weightsBQdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weightsBglobal_stepB:linear/linear_model/feature_14_embedding/embedding_weightsB:linear/linear_model/feature_18_embedding/embedding_weightsB9linear/linear_model/feature_2_embedding/embedding_weightsB9linear/linear_model/feature_7_embedding/embedding_weightsB9linear/linear_model/feature_8_embedding/embedding_weightsB9linear/linear_model/feature_9_embedding/embedding_weights*
dtype0*
_output_shapes
:
Я
save/SaveV2/shape_and_slicesConst"/device:CPU:0*я
valueхBтB B B B B109 25 0,109:0,25B32 19 0,32:0,19B149 25 0,149:0,25B4 5 0,4:0,5B29 19 0,29:0,19B165 25 0,165:0,25B B109 25 0,109:0,25B32 19 0,32:0,19B149 25 0,149:0,25B4 5 0,4:0,5B29 19 0,29:0,19B165 25 0,165:0,25*
dtype0*
_output_shapes
:

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices=dnn/hiddenlayer_0/batchnorm_0/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_0/batchnorm_0/moving_variance/Read/ReadVariableOp=dnn/hiddenlayer_1/batchnorm_1/moving_mean/Read/ReadVariableOpAdnn/hiddenlayer_1/batchnorm_1/moving_variance/Read/ReadVariableOp]dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/read]dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/read\dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/read\dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/read\dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/read\dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/readglobal_stepFlinear/linear_model/feature_14_embedding/embedding_weights/part_0/readFlinear/linear_model/feature_18_embedding/embedding_weights/part_0/readElinear/linear_model/feature_2_embedding/embedding_weights/part_0/readElinear/linear_model/feature_7_embedding/embedding_weights/part_0/readElinear/linear_model/feature_8_embedding/embedding_weights/part_0/readElinear/linear_model/feature_9_embedding/embedding_weights/part_0/read"/device:CPU:0*
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 

save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 

save/Read_28/ReadVariableOpReadVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0"/device:CPU:0*
_output_shapes
:*
dtype0
m
save/Identity_56Identitysave/Read_28/ReadVariableOp"/device:CPU:0*
_output_shapes
:*
T0
b
save/Identity_57Identitysave/Identity_56"/device:CPU:0*
T0*
_output_shapes
:

save/Read_29/ReadVariableOpReadVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0"/device:CPU:0*
_output_shapes
:*
dtype0
m
save/Identity_58Identitysave/Read_29/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_59Identitysave/Identity_58"/device:CPU:0*
T0*
_output_shapes
:

save/Read_30/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_60Identitysave/Read_30/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_61Identitysave/Identity_60"/device:CPU:0*
T0*
_output_shapes
:

save/Read_31/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes
:	
r
save/Identity_62Identitysave/Read_31/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	
g
save/Identity_63Identitysave/Identity_62"/device:CPU:0*
_output_shapes
:	*
T0

save/Read_32/ReadVariableOpReadVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0"/device:CPU:0*
_output_shapes
:*
dtype0
m
save/Identity_64Identitysave/Read_32/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_65Identitysave/Identity_64"/device:CPU:0*
T0*
_output_shapes
:

save/Read_33/ReadVariableOpReadVariableOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_66Identitysave/Read_33/ReadVariableOp"/device:CPU:0*
_output_shapes
:*
T0
b
save/Identity_67Identitysave/Identity_66"/device:CPU:0*
T0*
_output_shapes
:

save/Read_34/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_68Identitysave/Read_34/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_69Identitysave/Identity_68"/device:CPU:0*
T0*
_output_shapes
:

save/Read_35/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_70Identitysave/Read_35/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_71Identitysave/Identity_70"/device:CPU:0*
_output_shapes

:*
T0
}
save/Read_36/ReadVariableOpReadVariableOpdnn/logits/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_72Identitysave/Read_36/ReadVariableOp"/device:CPU:0*
_output_shapes
:*
T0
b
save/Identity_73Identitysave/Identity_72"/device:CPU:0*
T0*
_output_shapes
:

save/Read_37/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_74Identitysave/Read_37/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
f
save/Identity_75Identitysave/Identity_74"/device:CPU:0*
_output_shapes

:*
T0

save/Read_38/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_76Identitysave/Read_38/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_77Identitysave/Identity_76"/device:CPU:0*
T0*
_output_shapes
:

save/Read_39/ReadVariableOpReadVariableOp-linear/linear_model/feature_10/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_78Identitysave/Read_39/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
f
save/Identity_79Identitysave/Identity_78"/device:CPU:0*
_output_shapes

:*
T0

save/Read_40/ReadVariableOpReadVariableOp-linear/linear_model/feature_11/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_80Identitysave/Read_40/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
f
save/Identity_81Identitysave/Identity_80"/device:CPU:0*
T0*
_output_shapes

:

save/Read_41/ReadVariableOpReadVariableOp-linear/linear_model/feature_12/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_82Identitysave/Read_41/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_83Identitysave/Identity_82"/device:CPU:0*
T0*
_output_shapes

:

save/Read_42/ReadVariableOpReadVariableOp-linear/linear_model/feature_13/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_84Identitysave/Read_42/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
f
save/Identity_85Identitysave/Identity_84"/device:CPU:0*
T0*
_output_shapes

:
Ђ
save/Read_43/ReadVariableOpReadVariableOp7linear/linear_model/feature_14_embedding/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_86Identitysave/Read_43/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_87Identitysave/Identity_86"/device:CPU:0*
T0*
_output_shapes

:

save/Read_44/ReadVariableOpReadVariableOp-linear/linear_model/feature_15/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_88Identitysave/Read_44/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_89Identitysave/Identity_88"/device:CPU:0*
T0*
_output_shapes

:

save/Read_45/ReadVariableOpReadVariableOp-linear/linear_model/feature_16/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_90Identitysave/Read_45/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_91Identitysave/Identity_90"/device:CPU:0*
_output_shapes

:*
T0

save/Read_46/ReadVariableOpReadVariableOp-linear/linear_model/feature_17/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_92Identitysave/Read_46/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_93Identitysave/Identity_92"/device:CPU:0*
T0*
_output_shapes

:
Ђ
save/Read_47/ReadVariableOpReadVariableOp7linear/linear_model/feature_18_embedding/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_94Identitysave/Read_47/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_95Identitysave/Identity_94"/device:CPU:0*
_output_shapes

:*
T0
Ё
save/Read_48/ReadVariableOpReadVariableOp6linear/linear_model/feature_2_embedding/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_96Identitysave/Read_48/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_97Identitysave/Identity_96"/device:CPU:0*
T0*
_output_shapes

:

save/Read_49/ReadVariableOpReadVariableOp,linear/linear_model/feature_3/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_98Identitysave/Read_49/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
f
save/Identity_99Identitysave/Identity_98"/device:CPU:0*
_output_shapes

:*
T0

save/Read_50/ReadVariableOpReadVariableOp,linear/linear_model/feature_4/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
r
save/Identity_100Identitysave/Read_50/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
h
save/Identity_101Identitysave/Identity_100"/device:CPU:0*
T0*
_output_shapes

:

save/Read_51/ReadVariableOpReadVariableOp,linear/linear_model/feature_5/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
r
save/Identity_102Identitysave/Read_51/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
h
save/Identity_103Identitysave/Identity_102"/device:CPU:0*
T0*
_output_shapes

:

save/Read_52/ReadVariableOpReadVariableOp,linear/linear_model/feature_6/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
r
save/Identity_104Identitysave/Read_52/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
h
save/Identity_105Identitysave/Identity_104"/device:CPU:0*
T0*
_output_shapes

:
Ё
save/Read_53/ReadVariableOpReadVariableOp6linear/linear_model/feature_7_embedding/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
r
save/Identity_106Identitysave/Read_53/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
h
save/Identity_107Identitysave/Identity_106"/device:CPU:0*
T0*
_output_shapes

:
Ё
save/Read_54/ReadVariableOpReadVariableOp6linear/linear_model/feature_8_embedding/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
r
save/Identity_108Identitysave/Read_54/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
h
save/Identity_109Identitysave/Identity_108"/device:CPU:0*
T0*
_output_shapes

:
Ё
save/Read_55/ReadVariableOpReadVariableOp6linear/linear_model/feature_9_embedding/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
r
save/Identity_110Identitysave/Read_55/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
h
save/Identity_111Identitysave/Identity_110"/device:CPU:0*
_output_shapes

:*
T0
	
save/SaveV2_1/tensor_namesConst"/device:CPU:0*­
valueЃB B"dnn/hiddenlayer_0/batchnorm_0/betaB#dnn/hiddenlayer_0/batchnorm_0/gammaBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelB"dnn/hiddenlayer_1/batchnorm_1/betaB#dnn/hiddenlayer_1/batchnorm_1/gammaBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelB linear/linear_model/bias_weightsB&linear/linear_model/feature_10/weightsB&linear/linear_model/feature_11/weightsB&linear/linear_model/feature_12/weightsB&linear/linear_model/feature_13/weightsB0linear/linear_model/feature_14_embedding/weightsB&linear/linear_model/feature_15/weightsB&linear/linear_model/feature_16/weightsB&linear/linear_model/feature_17/weightsB0linear/linear_model/feature_18_embedding/weightsB/linear/linear_model/feature_2_embedding/weightsB%linear/linear_model/feature_3/weightsB%linear/linear_model/feature_4/weightsB%linear/linear_model/feature_5/weightsB%linear/linear_model/feature_6/weightsB/linear/linear_model/feature_7_embedding/weightsB/linear/linear_model/feature_8_embedding/weightsB/linear/linear_model/feature_9_embedding/weights*
dtype0*
_output_shapes
:
е
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*ѓ
valueщBцB16 0,16B16 0,16B16 0,16B129 16 0,129:0,16B16 0,16B16 0,16B16 0,16B16 16 0,16:0,16B1 0,1B16 1 0,16:0,1B1 0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B25 1 0,25:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B19 1 0,19:0,1B25 1 0,25:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B5 1 0,5:0,1B19 1 0,19:0,1B25 1 0,25:0,1*
dtype0*
_output_shapes
:
Є
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_57save/Identity_59save/Identity_61save/Identity_63save/Identity_65save/Identity_67save/Identity_69save/Identity_71save/Identity_73save/Identity_75save/Identity_77save/Identity_79save/Identity_81save/Identity_83save/Identity_85save/Identity_87save/Identity_89save/Identity_91save/Identity_93save/Identity_95save/Identity_97save/Identity_99save/Identity_101save/Identity_103save/Identity_105save/Identity_107save/Identity_109save/Identity_111"/device:CPU:0**
dtypes 
2
Ј
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
д
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
Љ
save/Identity_112Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
	
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*Ј
valueBB)dnn/hiddenlayer_0/batchnorm_0/moving_meanB-dnn/hiddenlayer_0/batchnorm_0/moving_varianceB)dnn/hiddenlayer_1/batchnorm_1/moving_meanB-dnn/hiddenlayer_1/batchnorm_1/moving_varianceBQdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weightsBQdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weightsBPdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weightsBglobal_stepB:linear/linear_model/feature_14_embedding/embedding_weightsB:linear/linear_model/feature_18_embedding/embedding_weightsB9linear/linear_model/feature_2_embedding/embedding_weightsB9linear/linear_model/feature_7_embedding/embedding_weightsB9linear/linear_model/feature_8_embedding/embedding_weightsB9linear/linear_model/feature_9_embedding/embedding_weights*
dtype0
в
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*я
valueхBтB B B B B109 25 0,109:0,25B32 19 0,32:0,19B149 25 0,149:0,25B4 5 0,4:0,5B29 19 0,29:0,19B165 25 0,165:0,25B B109 25 0,109:0,25B32 19 0,32:0,19B149 25 0,149:0,25B4 5 0,4:0,5B29 19 0,29:0,19B165 25 0,165:0,25*
dtype0*
_output_shapes
:
О
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*І
_output_shapes
:::::m: :	:::	Ѕ::m: :	:::	Ѕ*
dtypes
2	
P
save/Identity_113Identitysave/RestoreV2*
T0*
_output_shapes
:
t
save/AssignVariableOpAssignVariableOp)dnn/hiddenlayer_0/batchnorm_0/moving_meansave/Identity_113*
dtype0
R
save/Identity_114Identitysave/RestoreV2:1*
_output_shapes
:*
T0
z
save/AssignVariableOp_1AssignVariableOp-dnn/hiddenlayer_0/batchnorm_0/moving_variancesave/Identity_114*
dtype0
R
save/Identity_115Identitysave/RestoreV2:2*
_output_shapes
:*
T0
v
save/AssignVariableOp_2AssignVariableOp)dnn/hiddenlayer_1/batchnorm_1/moving_meansave/Identity_115*
dtype0
R
save/Identity_116Identitysave/RestoreV2:3*
_output_shapes
:*
T0
z
save/AssignVariableOp_3AssignVariableOp-dnn/hiddenlayer_1/batchnorm_1/moving_variancesave/Identity_116*
dtype0

save/AssignAssignXdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0save/RestoreV2:4*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0*
_output_shapes

:m

save/Assign_1AssignXdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0save/RestoreV2:5*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0*
_output_shapes

: 

save/Assign_2AssignWdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0save/RestoreV2:6*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	

save/Assign_3AssignWdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0save/RestoreV2:7*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0*
_output_shapes

:

save/Assign_4AssignWdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0save/RestoreV2:8*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0*
_output_shapes

:

save/Assign_5AssignWdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0save/RestoreV2:9*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0*
_output_shapes
:	Ѕ
x
save/Assign_6Assignglobal_stepsave/RestoreV2:10*
T0	*
_class
loc:@global_step*
_output_shapes
: 
ь
save/Assign_7AssignAlinear/linear_model/feature_14_embedding/embedding_weights/part_0save/RestoreV2:11*
T0*T
_classJ
HFloc:@linear/linear_model/feature_14_embedding/embedding_weights/part_0*
_output_shapes

:m
ь
save/Assign_8AssignAlinear/linear_model/feature_18_embedding/embedding_weights/part_0save/RestoreV2:12*
T0*T
_classJ
HFloc:@linear/linear_model/feature_18_embedding/embedding_weights/part_0*
_output_shapes

: 
ы
save/Assign_9Assign@linear/linear_model/feature_2_embedding/embedding_weights/part_0save/RestoreV2:13*
T0*S
_classI
GEloc:@linear/linear_model/feature_2_embedding/embedding_weights/part_0*
_output_shapes
:	
ы
save/Assign_10Assign@linear/linear_model/feature_7_embedding/embedding_weights/part_0save/RestoreV2:14*
T0*S
_classI
GEloc:@linear/linear_model/feature_7_embedding/embedding_weights/part_0*
_output_shapes

:
ы
save/Assign_11Assign@linear/linear_model/feature_8_embedding/embedding_weights/part_0save/RestoreV2:15*
T0*S
_classI
GEloc:@linear/linear_model/feature_8_embedding/embedding_weights/part_0*
_output_shapes

:
ь
save/Assign_12Assign@linear/linear_model/feature_9_embedding/embedding_weights/part_0save/RestoreV2:16*
_output_shapes
:	Ѕ*
T0*S
_classI
GEloc:@linear/linear_model/feature_9_embedding/embedding_weights/part_0
б
save/restore_shardNoOp^save/Assign^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
	
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*­
valueЃB B"dnn/hiddenlayer_0/batchnorm_0/betaB#dnn/hiddenlayer_0/batchnorm_0/gammaBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelB"dnn/hiddenlayer_1/batchnorm_1/betaB#dnn/hiddenlayer_1/batchnorm_1/gammaBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelB linear/linear_model/bias_weightsB&linear/linear_model/feature_10/weightsB&linear/linear_model/feature_11/weightsB&linear/linear_model/feature_12/weightsB&linear/linear_model/feature_13/weightsB0linear/linear_model/feature_14_embedding/weightsB&linear/linear_model/feature_15/weightsB&linear/linear_model/feature_16/weightsB&linear/linear_model/feature_17/weightsB0linear/linear_model/feature_18_embedding/weightsB/linear/linear_model/feature_2_embedding/weightsB%linear/linear_model/feature_3/weightsB%linear/linear_model/feature_4/weightsB%linear/linear_model/feature_5/weightsB%linear/linear_model/feature_6/weightsB/linear/linear_model/feature_7_embedding/weightsB/linear/linear_model/feature_8_embedding/weightsB/linear/linear_model/feature_9_embedding/weights*
dtype0*
_output_shapes
:
и
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*ѓ
valueщBцB16 0,16B16 0,16B16 0,16B129 16 0,129:0,16B16 0,16B16 0,16B16 0,16B16 16 0,16:0,16B1 0,1B16 1 0,16:0,1B1 0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B25 1 0,25:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B19 1 0,19:0,1B25 1 0,25:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B5 1 0,5:0,1B19 1 0,19:0,1B25 1 0,25:0,1*
dtype0*
_output_shapes
:
И
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*
_output_shapesќ
љ::::	::::::::::::::::::::::::**
dtypes 
2
c
save/Identity_117Identitysave/RestoreV2_1"/device:CPU:0*
T0*
_output_shapes
:

save/AssignVariableOp_4AssignVariableOp)dnn/hiddenlayer_0/batchnorm_0/beta/part_0save/Identity_117"/device:CPU:0*
dtype0
e
save/Identity_118Identitysave/RestoreV2_1:1"/device:CPU:0*
T0*
_output_shapes
:

save/AssignVariableOp_5AssignVariableOp*dnn/hiddenlayer_0/batchnorm_0/gamma/part_0save/Identity_118"/device:CPU:0*
dtype0
e
save/Identity_119Identitysave/RestoreV2_1:2"/device:CPU:0*
T0*
_output_shapes
:
y
save/AssignVariableOp_6AssignVariableOpdnn/hiddenlayer_0/bias/part_0save/Identity_119"/device:CPU:0*
dtype0
j
save/Identity_120Identitysave/RestoreV2_1:3"/device:CPU:0*
T0*
_output_shapes
:	
{
save/AssignVariableOp_7AssignVariableOpdnn/hiddenlayer_0/kernel/part_0save/Identity_120"/device:CPU:0*
dtype0
e
save/Identity_121Identitysave/RestoreV2_1:4"/device:CPU:0*
_output_shapes
:*
T0

save/AssignVariableOp_8AssignVariableOp)dnn/hiddenlayer_1/batchnorm_1/beta/part_0save/Identity_121"/device:CPU:0*
dtype0
e
save/Identity_122Identitysave/RestoreV2_1:5"/device:CPU:0*
T0*
_output_shapes
:

save/AssignVariableOp_9AssignVariableOp*dnn/hiddenlayer_1/batchnorm_1/gamma/part_0save/Identity_122"/device:CPU:0*
dtype0
e
save/Identity_123Identitysave/RestoreV2_1:6"/device:CPU:0*
T0*
_output_shapes
:
z
save/AssignVariableOp_10AssignVariableOpdnn/hiddenlayer_1/bias/part_0save/Identity_123"/device:CPU:0*
dtype0
i
save/Identity_124Identitysave/RestoreV2_1:7"/device:CPU:0*
T0*
_output_shapes

:
|
save/AssignVariableOp_11AssignVariableOpdnn/hiddenlayer_1/kernel/part_0save/Identity_124"/device:CPU:0*
dtype0
e
save/Identity_125Identitysave/RestoreV2_1:8"/device:CPU:0*
T0*
_output_shapes
:
s
save/AssignVariableOp_12AssignVariableOpdnn/logits/bias/part_0save/Identity_125"/device:CPU:0*
dtype0
i
save/Identity_126Identitysave/RestoreV2_1:9"/device:CPU:0*
T0*
_output_shapes

:
u
save/AssignVariableOp_13AssignVariableOpdnn/logits/kernel/part_0save/Identity_126"/device:CPU:0*
dtype0
f
save/Identity_127Identitysave/RestoreV2_1:10"/device:CPU:0*
T0*
_output_shapes
:

save/AssignVariableOp_14AssignVariableOp'linear/linear_model/bias_weights/part_0save/Identity_127"/device:CPU:0*
dtype0
j
save/Identity_128Identitysave/RestoreV2_1:11"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_15AssignVariableOp-linear/linear_model/feature_10/weights/part_0save/Identity_128"/device:CPU:0*
dtype0
j
save/Identity_129Identitysave/RestoreV2_1:12"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_16AssignVariableOp-linear/linear_model/feature_11/weights/part_0save/Identity_129"/device:CPU:0*
dtype0
j
save/Identity_130Identitysave/RestoreV2_1:13"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_17AssignVariableOp-linear/linear_model/feature_12/weights/part_0save/Identity_130"/device:CPU:0*
dtype0
j
save/Identity_131Identitysave/RestoreV2_1:14"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_18AssignVariableOp-linear/linear_model/feature_13/weights/part_0save/Identity_131"/device:CPU:0*
dtype0
j
save/Identity_132Identitysave/RestoreV2_1:15"/device:CPU:0*
_output_shapes

:*
T0

save/AssignVariableOp_19AssignVariableOp7linear/linear_model/feature_14_embedding/weights/part_0save/Identity_132"/device:CPU:0*
dtype0
j
save/Identity_133Identitysave/RestoreV2_1:16"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_20AssignVariableOp-linear/linear_model/feature_15/weights/part_0save/Identity_133"/device:CPU:0*
dtype0
j
save/Identity_134Identitysave/RestoreV2_1:17"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_21AssignVariableOp-linear/linear_model/feature_16/weights/part_0save/Identity_134"/device:CPU:0*
dtype0
j
save/Identity_135Identitysave/RestoreV2_1:18"/device:CPU:0*
_output_shapes

:*
T0

save/AssignVariableOp_22AssignVariableOp-linear/linear_model/feature_17/weights/part_0save/Identity_135"/device:CPU:0*
dtype0
j
save/Identity_136Identitysave/RestoreV2_1:19"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_23AssignVariableOp7linear/linear_model/feature_18_embedding/weights/part_0save/Identity_136"/device:CPU:0*
dtype0
j
save/Identity_137Identitysave/RestoreV2_1:20"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_24AssignVariableOp6linear/linear_model/feature_2_embedding/weights/part_0save/Identity_137"/device:CPU:0*
dtype0
j
save/Identity_138Identitysave/RestoreV2_1:21"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_25AssignVariableOp,linear/linear_model/feature_3/weights/part_0save/Identity_138"/device:CPU:0*
dtype0
j
save/Identity_139Identitysave/RestoreV2_1:22"/device:CPU:0*
_output_shapes

:*
T0

save/AssignVariableOp_26AssignVariableOp,linear/linear_model/feature_4/weights/part_0save/Identity_139"/device:CPU:0*
dtype0
j
save/Identity_140Identitysave/RestoreV2_1:23"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_27AssignVariableOp,linear/linear_model/feature_5/weights/part_0save/Identity_140"/device:CPU:0*
dtype0
j
save/Identity_141Identitysave/RestoreV2_1:24"/device:CPU:0*
_output_shapes

:*
T0

save/AssignVariableOp_28AssignVariableOp,linear/linear_model/feature_6/weights/part_0save/Identity_141"/device:CPU:0*
dtype0
j
save/Identity_142Identitysave/RestoreV2_1:25"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_29AssignVariableOp6linear/linear_model/feature_7_embedding/weights/part_0save/Identity_142"/device:CPU:0*
dtype0
j
save/Identity_143Identitysave/RestoreV2_1:26"/device:CPU:0*
_output_shapes

:*
T0

save/AssignVariableOp_30AssignVariableOp6linear/linear_model/feature_8_embedding/weights/part_0save/Identity_143"/device:CPU:0*
dtype0
j
save/Identity_144Identitysave/RestoreV2_1:27"/device:CPU:0*
_output_shapes

:*
T0

save/AssignVariableOp_31AssignVariableOp6linear/linear_model/feature_9_embedding/weights/part_0save/Identity_144"/device:CPU:0*
dtype0

save/restore_shard_1NoOp^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"@
save/Const:0save/Identity_112:0save/restore_all (5 @F8"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"чѕ	
cond_contextеѕ	бѕ	
к
 dnn/zero_fraction/cond/cond_text dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_t:0 *ю
/dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1:0
dnn/zero_fraction/cond/Cast:0
+dnn/zero_fraction/cond/count_nonzero/Cast:0
,dnn/zero_fraction/cond/count_nonzero/Const:0
6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
/dnn/zero_fraction/cond/count_nonzero/NotEqual:0
4dnn/zero_fraction/cond/count_nonzero/nonzero_count:0
,dnn/zero_fraction/cond/count_nonzero/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_t:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0i
/dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1:06dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Щ
"dnn/zero_fraction/cond/cond_text_1 dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_f:0*н
/dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1:0
-dnn/zero_fraction/cond/count_nonzero_1/Cast:0
.dnn/zero_fraction/cond/count_nonzero_1/Const:0
8dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
1dnn/zero_fraction/cond/count_nonzero_1/NotEqual:0
6dnn/zero_fraction/cond/count_nonzero_1/nonzero_count:0
.dnn/zero_fraction/cond/count_nonzero_1/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_f:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0k
/dnn/hiddenlayer_0/batchnorm_0/batchnorm/add_1:08dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
ј
"dnn/zero_fraction_1/cond/cond_text"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_t:0 *
/dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1:0
dnn/zero_fraction_1/cond/Cast:0
-dnn/zero_fraction_1/cond/count_nonzero/Cast:0
.dnn/zero_fraction_1/cond/count_nonzero/Const:0
8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_1/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_1/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_1/cond/count_nonzero/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_t:0k
/dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1:08dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0
х
$dnn/zero_fraction_1/cond/cond_text_1"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_f:0*ѓ
/dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1:0
/dnn/zero_fraction_1/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_1/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_1/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_f:0m
/dnn/hiddenlayer_1/batchnorm_1/batchnorm/add_1:0:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0
Т
"dnn/zero_fraction_2/cond/cond_text"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_t:0 *а
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
Џ
$dnn/zero_fraction_2/cond/cond_text_1"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_f:0*Н
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
dnn/logits/BiasAdd:0:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0
З
4linear/zero_fraction/total_zero/zero_count/cond_text4linear/zero_fraction/total_zero/zero_count/pred_id:05linear/zero_fraction/total_zero/zero_count/switch_t:0 *
2linear/zero_fraction/total_zero/zero_count/Const:0
4linear/zero_fraction/total_zero/zero_count/pred_id:0
5linear/zero_fraction/total_zero/zero_count/switch_t:0l
4linear/zero_fraction/total_zero/zero_count/pred_id:04linear/zero_fraction/total_zero/zero_count/pred_id:0
ъ.
6linear/zero_fraction/total_zero/zero_count/cond_text_14linear/zero_fraction/total_zero/zero_count/pred_id:05linear/zero_fraction/total_zero/zero_count/switch_f:0*М
/linear/linear_model/feature_10/weights/part_0:0
&linear/zero_fraction/total_size/Size:0
;linear/zero_fraction/total_zero/zero_count/ToFloat/Switch:0
4linear/zero_fraction/total_zero/zero_count/ToFloat:0
0linear/zero_fraction/total_zero/zero_count/mul:0
4linear/zero_fraction/total_zero/zero_count/pred_id:0
5linear/zero_fraction/total_zero/zero_count/switch_f:0
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/y:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual:0
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch:0
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
?linear/zero_fraction/total_zero/zero_count/zero_fraction/Size:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast:0
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge:0
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge:1
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:0
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:1
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Cast:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual:0
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Cast:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const:0
_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Xlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1:0
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/sub:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truediv:0
Clinear/zero_fraction/total_zero/zero_count/zero_fraction/fraction:0e
&linear/zero_fraction/total_size/Size:0;linear/zero_fraction/total_zero/zero_count/ToFloat/Switch:0
/linear/linear_model/feature_10/weights/part_0:0Plinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch:0l
4linear/zero_fraction/total_zero/zero_count/pred_id:04linear/zero_fraction/total_zero/zero_count/pred_id:02м

й

Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/cond_textGlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0 *ј
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast:0
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Cast:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual:0
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Њ
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:12Є

Ё

Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/cond_text_1Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0*Р
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Cast:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const:0
_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Xlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0Ќ
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Ч
6linear/zero_fraction/total_zero/zero_count_1/cond_text6linear/zero_fraction/total_zero/zero_count_1/pred_id:07linear/zero_fraction/total_zero/zero_count_1/switch_t:0 *
4linear/zero_fraction/total_zero/zero_count_1/Const:0
6linear/zero_fraction/total_zero/zero_count_1/pred_id:0
7linear/zero_fraction/total_zero/zero_count_1/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_1/pred_id:06linear/zero_fraction/total_zero/zero_count_1/pred_id:0
0
8linear/zero_fraction/total_zero/zero_count_1/cond_text_16linear/zero_fraction/total_zero/zero_count_1/pred_id:07linear/zero_fraction/total_zero/zero_count_1/switch_f:0*
/linear/linear_model/feature_11/weights/part_0:0
(linear/zero_fraction/total_size/Size_1:0
=linear/zero_fraction/total_zero/zero_count_1/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_1/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_1/mul:0
6linear/zero_fraction/total_zero/zero_count_1/pred_id:0
7linear/zero_fraction/total_zero/zero_count_1/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_1/zero_fraction/fraction:0i
(linear/zero_fraction/total_size/Size_1:0=linear/zero_fraction/total_zero/zero_count_1/ToFloat/Switch:0
/linear/linear_model/feature_11/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/Switch:0p
6linear/zero_fraction/total_zero/zero_count_1/pred_id:06linear/zero_fraction/total_zero/zero_count_1/pred_id:02ў

ћ

Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t:0 *	
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t:0Ў
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:02Ф

С

Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f:0*к
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f:0А
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Ч
6linear/zero_fraction/total_zero/zero_count_2/cond_text6linear/zero_fraction/total_zero/zero_count_2/pred_id:07linear/zero_fraction/total_zero/zero_count_2/switch_t:0 *
4linear/zero_fraction/total_zero/zero_count_2/Const:0
6linear/zero_fraction/total_zero/zero_count_2/pred_id:0
7linear/zero_fraction/total_zero/zero_count_2/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_2/pred_id:06linear/zero_fraction/total_zero/zero_count_2/pred_id:0
0
8linear/zero_fraction/total_zero/zero_count_2/cond_text_16linear/zero_fraction/total_zero/zero_count_2/pred_id:07linear/zero_fraction/total_zero/zero_count_2/switch_f:0*
/linear/linear_model/feature_12/weights/part_0:0
(linear/zero_fraction/total_size/Size_2:0
=linear/zero_fraction/total_zero/zero_count_2/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_2/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_2/mul:0
6linear/zero_fraction/total_zero/zero_count_2/pred_id:0
7linear/zero_fraction/total_zero/zero_count_2/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fraction:0i
(linear/zero_fraction/total_size/Size_2:0=linear/zero_fraction/total_zero/zero_count_2/ToFloat/Switch:0p
6linear/zero_fraction/total_zero/zero_count_2/pred_id:06linear/zero_fraction/total_zero/zero_count_2/pred_id:0
/linear/linear_model/feature_12/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/Switch:02ў

ћ

Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t:0 *	
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t:0Ў
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:02Ф

С

Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f:0*к
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f:0А
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Ч
6linear/zero_fraction/total_zero/zero_count_3/cond_text6linear/zero_fraction/total_zero/zero_count_3/pred_id:07linear/zero_fraction/total_zero/zero_count_3/switch_t:0 *
4linear/zero_fraction/total_zero/zero_count_3/Const:0
6linear/zero_fraction/total_zero/zero_count_3/pred_id:0
7linear/zero_fraction/total_zero/zero_count_3/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_3/pred_id:06linear/zero_fraction/total_zero/zero_count_3/pred_id:0
0
8linear/zero_fraction/total_zero/zero_count_3/cond_text_16linear/zero_fraction/total_zero/zero_count_3/pred_id:07linear/zero_fraction/total_zero/zero_count_3/switch_f:0*
/linear/linear_model/feature_13/weights/part_0:0
(linear/zero_fraction/total_size/Size_3:0
=linear/zero_fraction/total_zero/zero_count_3/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_3/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_3/mul:0
6linear/zero_fraction/total_zero/zero_count_3/pred_id:0
7linear/zero_fraction/total_zero/zero_count_3/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fraction:0
/linear/linear_model/feature_13/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/Switch:0p
6linear/zero_fraction/total_zero/zero_count_3/pred_id:06linear/zero_fraction/total_zero/zero_count_3/pred_id:0i
(linear/zero_fraction/total_size/Size_3:0=linear/zero_fraction/total_zero/zero_count_3/ToFloat/Switch:02ў

ћ

Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t:0 *	
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t:0Ў
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:02Ф

С

Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f:0*к
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f:0А
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Ч
6linear/zero_fraction/total_zero/zero_count_4/cond_text6linear/zero_fraction/total_zero/zero_count_4/pred_id:07linear/zero_fraction/total_zero/zero_count_4/switch_t:0 *
4linear/zero_fraction/total_zero/zero_count_4/Const:0
6linear/zero_fraction/total_zero/zero_count_4/pred_id:0
7linear/zero_fraction/total_zero/zero_count_4/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_4/pred_id:06linear/zero_fraction/total_zero/zero_count_4/pred_id:0
Щ4
8linear/zero_fraction/total_zero/zero_count_4/cond_text_16linear/zero_fraction/total_zero/zero_count_4/pred_id:07linear/zero_fraction/total_zero/zero_count_4/switch_f:0*
Hlinear/linear_model/feature_14_embedding/embedding_weights/part_0/read:0
(linear/zero_fraction/total_size/Size_4:0
=linear/zero_fraction/total_zero/zero_count_4/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_4/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_4/mul:0
6linear/zero_fraction/total_zero/zero_count_4/pred_id:0
7linear/zero_fraction/total_zero/zero_count_4/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual:0
Alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Xlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_4/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_4/pred_id:06linear/zero_fraction/total_zero/zero_count_4/pred_id:0Ћ
Hlinear/linear_model/feature_14_embedding/embedding_weights/part_0/read:0_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:0i
(linear/zero_fraction/total_size/Size_4:0=linear/zero_fraction/total_zero/zero_count_4/ToFloat/Switch:02Ђ

Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t:0 *И
Hlinear/linear_model/feature_14_embedding/embedding_weights/part_0/read:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Xlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t:0Т
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:0_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:0­
Hlinear/linear_model/feature_14_embedding/embedding_weights/part_0/read:0alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:02ф
с
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f:0*њ

Hlinear/linear_model/feature_14_embedding/embedding_weights/part_0/read:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f:0­
Hlinear/linear_model/feature_14_embedding/embedding_weights/part_0/read:0alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Т
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:0_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
Ч
6linear/zero_fraction/total_zero/zero_count_5/cond_text6linear/zero_fraction/total_zero/zero_count_5/pred_id:07linear/zero_fraction/total_zero/zero_count_5/switch_t:0 *
4linear/zero_fraction/total_zero/zero_count_5/Const:0
6linear/zero_fraction/total_zero/zero_count_5/pred_id:0
7linear/zero_fraction/total_zero/zero_count_5/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_5/pred_id:06linear/zero_fraction/total_zero/zero_count_5/pred_id:0
0
8linear/zero_fraction/total_zero/zero_count_5/cond_text_16linear/zero_fraction/total_zero/zero_count_5/pred_id:07linear/zero_fraction/total_zero/zero_count_5/switch_f:0*Ђ
9linear/linear_model/feature_14_embedding/weights/part_0:0
(linear/zero_fraction/total_size/Size_5:0
=linear/zero_fraction/total_zero/zero_count_5/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_5/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_5/mul:0
6linear/zero_fraction/total_zero/zero_count_5/pred_id:0
7linear/zero_fraction/total_zero/zero_count_5/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_5/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_5/pred_id:06linear/zero_fraction/total_zero/zero_count_5/pred_id:0
9linear/linear_model/feature_14_embedding/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/Switch:0i
(linear/zero_fraction/total_size/Size_5:0=linear/zero_fraction/total_zero/zero_count_5/ToFloat/Switch:02ў

ћ

Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t:0 *	
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t:0Ў
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:02Ф

С

Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f:0*к
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f:0А
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Ч
6linear/zero_fraction/total_zero/zero_count_6/cond_text6linear/zero_fraction/total_zero/zero_count_6/pred_id:07linear/zero_fraction/total_zero/zero_count_6/switch_t:0 *
4linear/zero_fraction/total_zero/zero_count_6/Const:0
6linear/zero_fraction/total_zero/zero_count_6/pred_id:0
7linear/zero_fraction/total_zero/zero_count_6/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_6/pred_id:06linear/zero_fraction/total_zero/zero_count_6/pred_id:0
0
8linear/zero_fraction/total_zero/zero_count_6/cond_text_16linear/zero_fraction/total_zero/zero_count_6/pred_id:07linear/zero_fraction/total_zero/zero_count_6/switch_f:0*
/linear/linear_model/feature_15/weights/part_0:0
(linear/zero_fraction/total_size/Size_6:0
=linear/zero_fraction/total_zero/zero_count_6/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_6/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_6/mul:0
6linear/zero_fraction/total_zero/zero_count_6/pred_id:0
7linear/zero_fraction/total_zero/zero_count_6/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_6/zero_fraction/fraction:0i
(linear/zero_fraction/total_size/Size_6:0=linear/zero_fraction/total_zero/zero_count_6/ToFloat/Switch:0
/linear/linear_model/feature_15/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/Switch:0p
6linear/zero_fraction/total_zero/zero_count_6/pred_id:06linear/zero_fraction/total_zero/zero_count_6/pred_id:02ў

ћ

Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t:0 *	
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t:0Ў
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:02Ф

С

Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f:0*к
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f:0А
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0
Ч
6linear/zero_fraction/total_zero/zero_count_7/cond_text6linear/zero_fraction/total_zero/zero_count_7/pred_id:07linear/zero_fraction/total_zero/zero_count_7/switch_t:0 *
4linear/zero_fraction/total_zero/zero_count_7/Const:0
6linear/zero_fraction/total_zero/zero_count_7/pred_id:0
7linear/zero_fraction/total_zero/zero_count_7/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_7/pred_id:06linear/zero_fraction/total_zero/zero_count_7/pred_id:0
0
8linear/zero_fraction/total_zero/zero_count_7/cond_text_16linear/zero_fraction/total_zero/zero_count_7/pred_id:07linear/zero_fraction/total_zero/zero_count_7/switch_f:0*
/linear/linear_model/feature_16/weights/part_0:0
(linear/zero_fraction/total_size/Size_7:0
=linear/zero_fraction/total_zero/zero_count_7/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_7/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_7/mul:0
6linear/zero_fraction/total_zero/zero_count_7/pred_id:0
7linear/zero_fraction/total_zero/zero_count_7/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_7/zero_fraction/fraction:0i
(linear/zero_fraction/total_size/Size_7:0=linear/zero_fraction/total_zero/zero_count_7/ToFloat/Switch:0p
6linear/zero_fraction/total_zero/zero_count_7/pred_id:06linear/zero_fraction/total_zero/zero_count_7/pred_id:0
/linear/linear_model/feature_16/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/Switch:02ў

ћ

Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t:0 *	
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t:0Ў
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:02Ф

С

Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f:0*к
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f:0А
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
Ч
6linear/zero_fraction/total_zero/zero_count_8/cond_text6linear/zero_fraction/total_zero/zero_count_8/pred_id:07linear/zero_fraction/total_zero/zero_count_8/switch_t:0 *
4linear/zero_fraction/total_zero/zero_count_8/Const:0
6linear/zero_fraction/total_zero/zero_count_8/pred_id:0
7linear/zero_fraction/total_zero/zero_count_8/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_8/pred_id:06linear/zero_fraction/total_zero/zero_count_8/pred_id:0
0
8linear/zero_fraction/total_zero/zero_count_8/cond_text_16linear/zero_fraction/total_zero/zero_count_8/pred_id:07linear/zero_fraction/total_zero/zero_count_8/switch_f:0*
/linear/linear_model/feature_17/weights/part_0:0
(linear/zero_fraction/total_size/Size_8:0
=linear/zero_fraction/total_zero/zero_count_8/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_8/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_8/mul:0
6linear/zero_fraction/total_zero/zero_count_8/pred_id:0
7linear/zero_fraction/total_zero/zero_count_8/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_8/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_8/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_8/pred_id:06linear/zero_fraction/total_zero/zero_count_8/pred_id:0i
(linear/zero_fraction/total_size/Size_8:0=linear/zero_fraction/total_zero/zero_count_8/ToFloat/Switch:0
/linear/linear_model/feature_17/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp/Switch:02ў

ћ

Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_t:0 *	
Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_t:0Ў
Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:02Ф

С

Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_f:0*к
Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_f:0А
Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0
Ч
6linear/zero_fraction/total_zero/zero_count_9/cond_text6linear/zero_fraction/total_zero/zero_count_9/pred_id:07linear/zero_fraction/total_zero/zero_count_9/switch_t:0 *
4linear/zero_fraction/total_zero/zero_count_9/Const:0
6linear/zero_fraction/total_zero/zero_count_9/pred_id:0
7linear/zero_fraction/total_zero/zero_count_9/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_9/pred_id:06linear/zero_fraction/total_zero/zero_count_9/pred_id:0
Щ4
8linear/zero_fraction/total_zero/zero_count_9/cond_text_16linear/zero_fraction/total_zero/zero_count_9/pred_id:07linear/zero_fraction/total_zero/zero_count_9/switch_f:0*
Hlinear/linear_model/feature_18_embedding/embedding_weights/part_0/read:0
(linear/zero_fraction/total_size/Size_9:0
=linear/zero_fraction/total_zero/zero_count_9/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_9/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_9/mul:0
6linear/zero_fraction/total_zero/zero_count_9/pred_id:0
7linear/zero_fraction/total_zero/zero_count_9/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual:0
Alinear/zero_fraction/total_zero/zero_count_9/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
alinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Xlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_9/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_9/pred_id:06linear/zero_fraction/total_zero/zero_count_9/pred_id:0Ћ
Hlinear/linear_model/feature_18_embedding/embedding_weights/part_0/read:0_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:0i
(linear/zero_fraction/total_size/Size_9:0=linear/zero_fraction/total_zero/zero_count_9/ToFloat/Switch:02Ђ

Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_t:0 *И
Hlinear/linear_model/feature_18_embedding/embedding_weights/part_0/read:0
Flinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
alinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Xlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_t:0
Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0­
Hlinear/linear_model/feature_18_embedding/embedding_weights/part_0/read:0alinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1Т
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:0_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:02ф
с
Klinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_f:0*њ

Hlinear/linear_model/feature_18_embedding/embedding_weights/part_0/read:0
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
Vlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0­
Hlinear/linear_model/feature_18_embedding/embedding_weights/part_0/read:0alinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0Т
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:0_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
Я
7linear/zero_fraction/total_zero/zero_count_10/cond_text7linear/zero_fraction/total_zero/zero_count_10/pred_id:08linear/zero_fraction/total_zero/zero_count_10/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_10/Const:0
7linear/zero_fraction/total_zero/zero_count_10/pred_id:0
8linear/zero_fraction/total_zero/zero_count_10/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_10/pred_id:07linear/zero_fraction/total_zero/zero_count_10/pred_id:0
х0
9linear/zero_fraction/total_zero/zero_count_10/cond_text_17linear/zero_fraction/total_zero/zero_count_10/pred_id:08linear/zero_fraction/total_zero/zero_count_10/switch_f:0*Ы
9linear/linear_model/feature_18_embedding/weights/part_0:0
)linear/zero_fraction/total_size/Size_10:0
>linear/zero_fraction/total_zero/zero_count_10/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_10/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_10/mul:0
7linear/zero_fraction/total_zero/zero_count_10/pred_id:0
8linear/zero_fraction/total_zero/zero_count_10/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual:0
Slinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp/Switch:0
Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp:0
Blinear/zero_fraction/total_zero/zero_count_10/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_10/zero_fraction/fraction:0r
7linear/zero_fraction/total_zero/zero_count_10/pred_id:07linear/zero_fraction/total_zero/zero_count_10/pred_id:0
9linear/linear_model/feature_18_embedding/weights/part_0:0Slinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp/Switch:0k
)linear/zero_fraction/total_size/Size_10:0>linear/zero_fraction/total_zero/zero_count_10/ToFloat/Switch:02

Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_t:0 *Ђ	
Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp:0
Glinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_t:0А
Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp:0`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:02д

б

Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_f:0*ч
Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp:0
Wlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_f:0В
Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp:0blinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0
Я
7linear/zero_fraction/total_zero/zero_count_11/cond_text7linear/zero_fraction/total_zero/zero_count_11/pred_id:08linear/zero_fraction/total_zero/zero_count_11/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_11/Const:0
7linear/zero_fraction/total_zero/zero_count_11/pred_id:0
8linear/zero_fraction/total_zero/zero_count_11/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_11/pred_id:07linear/zero_fraction/total_zero/zero_count_11/pred_id:0
5
9linear/zero_fraction/total_zero/zero_count_11/cond_text_17linear/zero_fraction/total_zero/zero_count_11/pred_id:08linear/zero_fraction/total_zero/zero_count_11/switch_f:0*Е
Glinear/linear_model/feature_2_embedding/embedding_weights/part_0/read:0
)linear/zero_fraction/total_size/Size_11:0
>linear/zero_fraction/total_zero/zero_count_11/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_11/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_11/mul:0
7linear/zero_fraction/total_zero/zero_count_11/pred_id:0
8linear/zero_fraction/total_zero/zero_count_11/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_11/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_11/zero_fraction/LessEqual:0
Blinear/zero_fraction/total_zero/zero_count_11/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
blinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Ylinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_11/zero_fraction/fraction:0k
)linear/zero_fraction/total_size/Size_11:0>linear/zero_fraction/total_zero/zero_count_11/ToFloat/Switch:0r
7linear/zero_fraction/total_zero/zero_count_11/pred_id:07linear/zero_fraction/total_zero/zero_count_11/pred_id:0Ћ
Glinear/linear_model/feature_2_embedding/embedding_weights/part_0/read:0`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch:02В
Џ
Jlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_t:0 *Х
Glinear/linear_model/feature_2_embedding/embedding_weights/part_0/read:0
Glinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
blinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Ylinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_t:0
Jlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_id:0Ф
`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch:0`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch:0­
Glinear/linear_model/feature_2_embedding/embedding_weights/part_0/read:0blinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:12ѓ
№
Llinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_f:0*
Glinear/linear_model/feature_2_embedding/embedding_weights/part_0/read:0
`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
Wlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/pred_id:0Ф
`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch:0`linear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero/NotEqual/Switch:0­
Glinear/linear_model/feature_2_embedding/embedding_weights/part_0/read:0blinear/zero_fraction/total_zero/zero_count_11/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Я
7linear/zero_fraction/total_zero/zero_count_12/cond_text7linear/zero_fraction/total_zero/zero_count_12/pred_id:08linear/zero_fraction/total_zero/zero_count_12/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_12/Const:0
7linear/zero_fraction/total_zero/zero_count_12/pred_id:0
8linear/zero_fraction/total_zero/zero_count_12/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_12/pred_id:07linear/zero_fraction/total_zero/zero_count_12/pred_id:0
у0
9linear/zero_fraction/total_zero/zero_count_12/cond_text_17linear/zero_fraction/total_zero/zero_count_12/pred_id:08linear/zero_fraction/total_zero/zero_count_12/switch_f:0*Щ
8linear/linear_model/feature_2_embedding/weights/part_0:0
)linear/zero_fraction/total_size/Size_12:0
>linear/zero_fraction/total_zero/zero_count_12/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_12/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_12/mul:0
7linear/zero_fraction/total_zero/zero_count_12/pred_id:0
8linear/zero_fraction/total_zero/zero_count_12/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_12/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_12/zero_fraction/LessEqual:0
Slinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOp/Switch:0
Llinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOp:0
Blinear/zero_fraction/total_zero/zero_count_12/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_12/zero_fraction/fraction:0k
)linear/zero_fraction/total_size/Size_12:0>linear/zero_fraction/total_zero/zero_count_12/ToFloat/Switch:0
8linear/linear_model/feature_2_embedding/weights/part_0:0Slinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOp/Switch:0r
7linear/zero_fraction/total_zero/zero_count_12/pred_id:07linear/zero_fraction/total_zero/zero_count_12/pred_id:02

Jlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_t:0 *Ђ	
Llinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOp:0
Glinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_t:0А
Llinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOp:0`linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Jlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_id:02д

б

Llinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_f:0*ч
Llinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOp:0
Wlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/switch_f:0В
Llinear/zero_fraction/total_zero/zero_count_12/zero_fraction/ReadVariableOp:0blinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Jlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_12/zero_fraction/cond/pred_id:0
Я
7linear/zero_fraction/total_zero/zero_count_13/cond_text7linear/zero_fraction/total_zero/zero_count_13/pred_id:08linear/zero_fraction/total_zero/zero_count_13/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_13/Const:0
7linear/zero_fraction/total_zero/zero_count_13/pred_id:0
8linear/zero_fraction/total_zero/zero_count_13/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_13/pred_id:07linear/zero_fraction/total_zero/zero_count_13/pred_id:0
Я0
9linear/zero_fraction/total_zero/zero_count_13/cond_text_17linear/zero_fraction/total_zero/zero_count_13/pred_id:08linear/zero_fraction/total_zero/zero_count_13/switch_f:0*Е
.linear/linear_model/feature_3/weights/part_0:0
)linear/zero_fraction/total_size/Size_13:0
>linear/zero_fraction/total_zero/zero_count_13/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_13/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_13/mul:0
7linear/zero_fraction/total_zero/zero_count_13/pred_id:0
8linear/zero_fraction/total_zero/zero_count_13/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_13/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_13/zero_fraction/LessEqual:0
Slinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOp/Switch:0
Llinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOp:0
Blinear/zero_fraction/total_zero/zero_count_13/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_13/zero_fraction/fraction:0r
7linear/zero_fraction/total_zero/zero_count_13/pred_id:07linear/zero_fraction/total_zero/zero_count_13/pred_id:0
.linear/linear_model/feature_3/weights/part_0:0Slinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOp/Switch:0k
)linear/zero_fraction/total_size/Size_13:0>linear/zero_fraction/total_zero/zero_count_13/ToFloat/Switch:02

Jlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_t:0 *Ђ	
Llinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOp:0
Glinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_t:0
Jlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_id:0А
Llinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOp:0`linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero/NotEqual/Switch:12д

б

Llinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_f:0*ч
Llinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOp:0
Wlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/switch_f:0В
Llinear/zero_fraction/total_zero/zero_count_13/zero_fraction/ReadVariableOp:0blinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Jlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_13/zero_fraction/cond/pred_id:0
Я
7linear/zero_fraction/total_zero/zero_count_14/cond_text7linear/zero_fraction/total_zero/zero_count_14/pred_id:08linear/zero_fraction/total_zero/zero_count_14/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_14/Const:0
7linear/zero_fraction/total_zero/zero_count_14/pred_id:0
8linear/zero_fraction/total_zero/zero_count_14/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_14/pred_id:07linear/zero_fraction/total_zero/zero_count_14/pred_id:0
Я0
9linear/zero_fraction/total_zero/zero_count_14/cond_text_17linear/zero_fraction/total_zero/zero_count_14/pred_id:08linear/zero_fraction/total_zero/zero_count_14/switch_f:0*Е
.linear/linear_model/feature_4/weights/part_0:0
)linear/zero_fraction/total_size/Size_14:0
>linear/zero_fraction/total_zero/zero_count_14/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_14/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_14/mul:0
7linear/zero_fraction/total_zero/zero_count_14/pred_id:0
8linear/zero_fraction/total_zero/zero_count_14/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_14/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_14/zero_fraction/LessEqual:0
Slinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOp/Switch:0
Llinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOp:0
Blinear/zero_fraction/total_zero/zero_count_14/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_14/zero_fraction/fraction:0
.linear/linear_model/feature_4/weights/part_0:0Slinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOp/Switch:0k
)linear/zero_fraction/total_size/Size_14:0>linear/zero_fraction/total_zero/zero_count_14/ToFloat/Switch:0r
7linear/zero_fraction/total_zero/zero_count_14/pred_id:07linear/zero_fraction/total_zero/zero_count_14/pred_id:02

Jlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_t:0 *Ђ	
Llinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOp:0
Glinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_t:0А
Llinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOp:0`linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Jlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_id:02д

б

Llinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_f:0*ч
Llinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOp:0
Wlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/switch_f:0В
Llinear/zero_fraction/total_zero/zero_count_14/zero_fraction/ReadVariableOp:0blinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Jlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_14/zero_fraction/cond/pred_id:0
Я
7linear/zero_fraction/total_zero/zero_count_15/cond_text7linear/zero_fraction/total_zero/zero_count_15/pred_id:08linear/zero_fraction/total_zero/zero_count_15/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_15/Const:0
7linear/zero_fraction/total_zero/zero_count_15/pred_id:0
8linear/zero_fraction/total_zero/zero_count_15/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_15/pred_id:07linear/zero_fraction/total_zero/zero_count_15/pred_id:0
Я0
9linear/zero_fraction/total_zero/zero_count_15/cond_text_17linear/zero_fraction/total_zero/zero_count_15/pred_id:08linear/zero_fraction/total_zero/zero_count_15/switch_f:0*Е
.linear/linear_model/feature_5/weights/part_0:0
)linear/zero_fraction/total_size/Size_15:0
>linear/zero_fraction/total_zero/zero_count_15/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_15/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_15/mul:0
7linear/zero_fraction/total_zero/zero_count_15/pred_id:0
8linear/zero_fraction/total_zero/zero_count_15/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_15/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_15/zero_fraction/LessEqual:0
Slinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOp/Switch:0
Llinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOp:0
Blinear/zero_fraction/total_zero/zero_count_15/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_15/zero_fraction/fraction:0k
)linear/zero_fraction/total_size/Size_15:0>linear/zero_fraction/total_zero/zero_count_15/ToFloat/Switch:0r
7linear/zero_fraction/total_zero/zero_count_15/pred_id:07linear/zero_fraction/total_zero/zero_count_15/pred_id:0
.linear/linear_model/feature_5/weights/part_0:0Slinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOp/Switch:02

Jlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_t:0 *Ђ	
Llinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOp:0
Glinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_t:0А
Llinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOp:0`linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Jlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_id:02д

б

Llinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_f:0*ч
Llinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOp:0
Wlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/switch_f:0В
Llinear/zero_fraction/total_zero/zero_count_15/zero_fraction/ReadVariableOp:0blinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Jlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_15/zero_fraction/cond/pred_id:0
Я
7linear/zero_fraction/total_zero/zero_count_16/cond_text7linear/zero_fraction/total_zero/zero_count_16/pred_id:08linear/zero_fraction/total_zero/zero_count_16/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_16/Const:0
7linear/zero_fraction/total_zero/zero_count_16/pred_id:0
8linear/zero_fraction/total_zero/zero_count_16/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_16/pred_id:07linear/zero_fraction/total_zero/zero_count_16/pred_id:0
Я0
9linear/zero_fraction/total_zero/zero_count_16/cond_text_17linear/zero_fraction/total_zero/zero_count_16/pred_id:08linear/zero_fraction/total_zero/zero_count_16/switch_f:0*Е
.linear/linear_model/feature_6/weights/part_0:0
)linear/zero_fraction/total_size/Size_16:0
>linear/zero_fraction/total_zero/zero_count_16/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_16/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_16/mul:0
7linear/zero_fraction/total_zero/zero_count_16/pred_id:0
8linear/zero_fraction/total_zero/zero_count_16/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_16/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_16/zero_fraction/LessEqual:0
Slinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOp/Switch:0
Llinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOp:0
Blinear/zero_fraction/total_zero/zero_count_16/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_16/zero_fraction/fraction:0r
7linear/zero_fraction/total_zero/zero_count_16/pred_id:07linear/zero_fraction/total_zero/zero_count_16/pred_id:0
.linear/linear_model/feature_6/weights/part_0:0Slinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOp/Switch:0k
)linear/zero_fraction/total_size/Size_16:0>linear/zero_fraction/total_zero/zero_count_16/ToFloat/Switch:02

Jlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_t:0 *Ђ	
Llinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOp:0
Glinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_t:0
Jlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_id:0А
Llinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOp:0`linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero/NotEqual/Switch:12д

б

Llinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_f:0*ч
Llinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOp:0
Wlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/switch_f:0В
Llinear/zero_fraction/total_zero/zero_count_16/zero_fraction/ReadVariableOp:0blinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Jlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_16/zero_fraction/cond/pred_id:0
Я
7linear/zero_fraction/total_zero/zero_count_17/cond_text7linear/zero_fraction/total_zero/zero_count_17/pred_id:08linear/zero_fraction/total_zero/zero_count_17/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_17/Const:0
7linear/zero_fraction/total_zero/zero_count_17/pred_id:0
8linear/zero_fraction/total_zero/zero_count_17/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_17/pred_id:07linear/zero_fraction/total_zero/zero_count_17/pred_id:0
5
9linear/zero_fraction/total_zero/zero_count_17/cond_text_17linear/zero_fraction/total_zero/zero_count_17/pred_id:08linear/zero_fraction/total_zero/zero_count_17/switch_f:0*Е
Glinear/linear_model/feature_7_embedding/embedding_weights/part_0/read:0
)linear/zero_fraction/total_size/Size_17:0
>linear/zero_fraction/total_zero/zero_count_17/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_17/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_17/mul:0
7linear/zero_fraction/total_zero/zero_count_17/pred_id:0
8linear/zero_fraction/total_zero/zero_count_17/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_17/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_17/zero_fraction/LessEqual:0
Blinear/zero_fraction/total_zero/zero_count_17/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
blinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Ylinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_17/zero_fraction/fraction:0Ћ
Glinear/linear_model/feature_7_embedding/embedding_weights/part_0/read:0`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch:0r
7linear/zero_fraction/total_zero/zero_count_17/pred_id:07linear/zero_fraction/total_zero/zero_count_17/pred_id:0k
)linear/zero_fraction/total_size/Size_17:0>linear/zero_fraction/total_zero/zero_count_17/ToFloat/Switch:02В
Џ
Jlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_t:0 *Х
Glinear/linear_model/feature_7_embedding/embedding_weights/part_0/read:0
Glinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
blinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Ylinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_t:0
Jlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_id:0Ф
`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch:0`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch:0­
Glinear/linear_model/feature_7_embedding/embedding_weights/part_0/read:0blinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:12ѓ
№
Llinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_f:0*
Glinear/linear_model/feature_7_embedding/embedding_weights/part_0/read:0
`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
Wlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/switch_f:0­
Glinear/linear_model/feature_7_embedding/embedding_weights/part_0/read:0blinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Jlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/pred_id:0Ф
`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch:0`linear/zero_fraction/total_zero/zero_count_17/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
Я
7linear/zero_fraction/total_zero/zero_count_18/cond_text7linear/zero_fraction/total_zero/zero_count_18/pred_id:08linear/zero_fraction/total_zero/zero_count_18/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_18/Const:0
7linear/zero_fraction/total_zero/zero_count_18/pred_id:0
8linear/zero_fraction/total_zero/zero_count_18/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_18/pred_id:07linear/zero_fraction/total_zero/zero_count_18/pred_id:0
у0
9linear/zero_fraction/total_zero/zero_count_18/cond_text_17linear/zero_fraction/total_zero/zero_count_18/pred_id:08linear/zero_fraction/total_zero/zero_count_18/switch_f:0*Щ
8linear/linear_model/feature_7_embedding/weights/part_0:0
)linear/zero_fraction/total_size/Size_18:0
>linear/zero_fraction/total_zero/zero_count_18/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_18/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_18/mul:0
7linear/zero_fraction/total_zero/zero_count_18/pred_id:0
8linear/zero_fraction/total_zero/zero_count_18/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_18/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_18/zero_fraction/LessEqual:0
Slinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOp/Switch:0
Llinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOp:0
Blinear/zero_fraction/total_zero/zero_count_18/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_18/zero_fraction/fraction:0
8linear/linear_model/feature_7_embedding/weights/part_0:0Slinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOp/Switch:0k
)linear/zero_fraction/total_size/Size_18:0>linear/zero_fraction/total_zero/zero_count_18/ToFloat/Switch:0r
7linear/zero_fraction/total_zero/zero_count_18/pred_id:07linear/zero_fraction/total_zero/zero_count_18/pred_id:02

Jlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_t:0 *Ђ	
Llinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOp:0
Glinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_t:0А
Llinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOp:0`linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Jlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_id:02д

б

Llinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_f:0*ч
Llinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOp:0
Wlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/switch_f:0В
Llinear/zero_fraction/total_zero/zero_count_18/zero_fraction/ReadVariableOp:0blinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Jlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_18/zero_fraction/cond/pred_id:0
Я
7linear/zero_fraction/total_zero/zero_count_19/cond_text7linear/zero_fraction/total_zero/zero_count_19/pred_id:08linear/zero_fraction/total_zero/zero_count_19/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_19/Const:0
7linear/zero_fraction/total_zero/zero_count_19/pred_id:0
8linear/zero_fraction/total_zero/zero_count_19/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_19/pred_id:07linear/zero_fraction/total_zero/zero_count_19/pred_id:0
5
9linear/zero_fraction/total_zero/zero_count_19/cond_text_17linear/zero_fraction/total_zero/zero_count_19/pred_id:08linear/zero_fraction/total_zero/zero_count_19/switch_f:0*Е
Glinear/linear_model/feature_8_embedding/embedding_weights/part_0/read:0
)linear/zero_fraction/total_size/Size_19:0
>linear/zero_fraction/total_zero/zero_count_19/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_19/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_19/mul:0
7linear/zero_fraction/total_zero/zero_count_19/pred_id:0
8linear/zero_fraction/total_zero/zero_count_19/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_19/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_19/zero_fraction/LessEqual:0
Blinear/zero_fraction/total_zero/zero_count_19/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
blinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Ylinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_19/zero_fraction/fraction:0Ћ
Glinear/linear_model/feature_8_embedding/embedding_weights/part_0/read:0`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch:0r
7linear/zero_fraction/total_zero/zero_count_19/pred_id:07linear/zero_fraction/total_zero/zero_count_19/pred_id:0k
)linear/zero_fraction/total_size/Size_19:0>linear/zero_fraction/total_zero/zero_count_19/ToFloat/Switch:02В
Џ
Jlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_t:0 *Х
Glinear/linear_model/feature_8_embedding/embedding_weights/part_0/read:0
Glinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
blinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Ylinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_t:0­
Glinear/linear_model/feature_8_embedding/embedding_weights/part_0/read:0blinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Jlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_id:0Ф
`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch:0`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch:02ѓ
№
Llinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_f:0*
Glinear/linear_model/feature_8_embedding/embedding_weights/part_0/read:0
`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
Wlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/pred_id:0Ф
`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch:0`linear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero/NotEqual/Switch:0­
Glinear/linear_model/feature_8_embedding/embedding_weights/part_0/read:0blinear/zero_fraction/total_zero/zero_count_19/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Я
7linear/zero_fraction/total_zero/zero_count_20/cond_text7linear/zero_fraction/total_zero/zero_count_20/pred_id:08linear/zero_fraction/total_zero/zero_count_20/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_20/Const:0
7linear/zero_fraction/total_zero/zero_count_20/pred_id:0
8linear/zero_fraction/total_zero/zero_count_20/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_20/pred_id:07linear/zero_fraction/total_zero/zero_count_20/pred_id:0
у0
9linear/zero_fraction/total_zero/zero_count_20/cond_text_17linear/zero_fraction/total_zero/zero_count_20/pred_id:08linear/zero_fraction/total_zero/zero_count_20/switch_f:0*Щ
8linear/linear_model/feature_8_embedding/weights/part_0:0
)linear/zero_fraction/total_size/Size_20:0
>linear/zero_fraction/total_zero/zero_count_20/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_20/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_20/mul:0
7linear/zero_fraction/total_zero/zero_count_20/pred_id:0
8linear/zero_fraction/total_zero/zero_count_20/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_20/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_20/zero_fraction/LessEqual:0
Slinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOp/Switch:0
Llinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOp:0
Blinear/zero_fraction/total_zero/zero_count_20/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_20/zero_fraction/fraction:0k
)linear/zero_fraction/total_size/Size_20:0>linear/zero_fraction/total_zero/zero_count_20/ToFloat/Switch:0r
7linear/zero_fraction/total_zero/zero_count_20/pred_id:07linear/zero_fraction/total_zero/zero_count_20/pred_id:0
8linear/linear_model/feature_8_embedding/weights/part_0:0Slinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOp/Switch:02

Jlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_t:0 *Ђ	
Llinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOp:0
Glinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_t:0А
Llinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOp:0`linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Jlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_id:02д

б

Llinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_f:0*ч
Llinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOp:0
Wlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/switch_f:0В
Llinear/zero_fraction/total_zero/zero_count_20/zero_fraction/ReadVariableOp:0blinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Jlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_20/zero_fraction/cond/pred_id:0
Я
7linear/zero_fraction/total_zero/zero_count_21/cond_text7linear/zero_fraction/total_zero/zero_count_21/pred_id:08linear/zero_fraction/total_zero/zero_count_21/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_21/Const:0
7linear/zero_fraction/total_zero/zero_count_21/pred_id:0
8linear/zero_fraction/total_zero/zero_count_21/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_21/pred_id:07linear/zero_fraction/total_zero/zero_count_21/pred_id:0
5
9linear/zero_fraction/total_zero/zero_count_21/cond_text_17linear/zero_fraction/total_zero/zero_count_21/pred_id:08linear/zero_fraction/total_zero/zero_count_21/switch_f:0*Е
Glinear/linear_model/feature_9_embedding/embedding_weights/part_0/read:0
)linear/zero_fraction/total_size/Size_21:0
>linear/zero_fraction/total_zero/zero_count_21/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_21/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_21/mul:0
7linear/zero_fraction/total_zero/zero_count_21/pred_id:0
8linear/zero_fraction/total_zero/zero_count_21/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_21/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_21/zero_fraction/LessEqual:0
Blinear/zero_fraction/total_zero/zero_count_21/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
blinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Ylinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_21/zero_fraction/fraction:0Ћ
Glinear/linear_model/feature_9_embedding/embedding_weights/part_0/read:0`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch:0k
)linear/zero_fraction/total_size/Size_21:0>linear/zero_fraction/total_zero/zero_count_21/ToFloat/Switch:0r
7linear/zero_fraction/total_zero/zero_count_21/pred_id:07linear/zero_fraction/total_zero/zero_count_21/pred_id:02В
Џ
Jlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_t:0 *Х
Glinear/linear_model/feature_9_embedding/embedding_weights/part_0/read:0
Glinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
blinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Ylinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_t:0­
Glinear/linear_model/feature_9_embedding/embedding_weights/part_0/read:0blinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch_1:1
Jlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_id:0Ф
`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch:0`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch:02ѓ
№
Llinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_f:0*
Glinear/linear_model/feature_9_embedding/embedding_weights/part_0/read:0
`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch:0
Wlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/switch_f:0Ф
`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch:0`linear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero/NotEqual/Switch:0­
Glinear/linear_model/feature_9_embedding/embedding_weights/part_0/read:0blinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Jlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_21/zero_fraction/cond/pred_id:0
Я
7linear/zero_fraction/total_zero/zero_count_22/cond_text7linear/zero_fraction/total_zero/zero_count_22/pred_id:08linear/zero_fraction/total_zero/zero_count_22/switch_t:0 *
5linear/zero_fraction/total_zero/zero_count_22/Const:0
7linear/zero_fraction/total_zero/zero_count_22/pred_id:0
8linear/zero_fraction/total_zero/zero_count_22/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_22/pred_id:07linear/zero_fraction/total_zero/zero_count_22/pred_id:0
у0
9linear/zero_fraction/total_zero/zero_count_22/cond_text_17linear/zero_fraction/total_zero/zero_count_22/pred_id:08linear/zero_fraction/total_zero/zero_count_22/switch_f:0*Щ
8linear/linear_model/feature_9_embedding/weights/part_0:0
)linear/zero_fraction/total_size/Size_22:0
>linear/zero_fraction/total_zero/zero_count_22/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_22/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_22/mul:0
7linear/zero_fraction/total_zero/zero_count_22/pred_id:0
8linear/zero_fraction/total_zero/zero_count_22/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_22/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_22/zero_fraction/LessEqual:0
Slinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOp/Switch:0
Llinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOp:0
Blinear/zero_fraction/total_zero/zero_count_22/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_22/zero_fraction/fraction:0
8linear/linear_model/feature_9_embedding/weights/part_0:0Slinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOp/Switch:0k
)linear/zero_fraction/total_size/Size_22:0>linear/zero_fraction/total_zero/zero_count_22/ToFloat/Switch:0r
7linear/zero_fraction/total_zero/zero_count_22/pred_id:07linear/zero_fraction/total_zero/zero_count_22/pred_id:02

Jlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_t:0 *Ђ	
Llinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOp:0
Glinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_t:0А
Llinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOp:0`linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Jlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_id:02д

б

Llinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_f:0*ч
Llinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOp:0
Wlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/switch_f:0В
Llinear/zero_fraction/total_zero/zero_count_22/zero_fraction/ReadVariableOp:0blinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Jlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_22/zero_fraction/cond/pred_id:0
П
%linear/zero_fraction_1/cond/cond_text%linear/zero_fraction_1/cond/pred_id:0&linear/zero_fraction_1/cond/switch_t:0 *Ф
<linear/linear_model/linear_model/linear_model/weighted_sum:0
"linear/zero_fraction_1/cond/Cast:0
0linear/zero_fraction_1/cond/count_nonzero/Cast:0
1linear/zero_fraction_1/cond/count_nonzero/Const:0
;linear/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
4linear/zero_fraction_1/cond/count_nonzero/NotEqual:0
9linear/zero_fraction_1/cond/count_nonzero/nonzero_count:0
1linear/zero_fraction_1/cond/count_nonzero/zeros:0
%linear/zero_fraction_1/cond/pred_id:0
&linear/zero_fraction_1/cond/switch_t:0N
%linear/zero_fraction_1/cond/pred_id:0%linear/zero_fraction_1/cond/pred_id:0{
<linear/linear_model/linear_model/linear_model/weighted_sum:0;linear/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
Љ
'linear/zero_fraction_1/cond/cond_text_1%linear/zero_fraction_1/cond/pred_id:0&linear/zero_fraction_1/cond/switch_f:0*Ў
<linear/linear_model/linear_model/linear_model/weighted_sum:0
2linear/zero_fraction_1/cond/count_nonzero_1/Cast:0
3linear/zero_fraction_1/cond/count_nonzero_1/Const:0
=linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
6linear/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
;linear/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
3linear/zero_fraction_1/cond/count_nonzero_1/zeros:0
%linear/zero_fraction_1/cond/pred_id:0
&linear/zero_fraction_1/cond/switch_f:0N
%linear/zero_fraction_1/cond/pred_id:0%linear/zero_fraction_1/cond/pred_id:0}
<linear/linear_model/linear_model/linear_model/weighted_sum:0=linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0"%
saved_model_main_op


group_deps"i
trainable_variablesјhѕh
њ
Zdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weightsm  "m2wdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
њ
Zdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights   " 2wdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ї
Ydnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/read:0"`
Pdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights  "2vdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ѕ
Ydnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/read:0"^
Pdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights  "2vdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ѕ
Ydnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/read:0"^
Pdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights  "2vdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ї
Ydnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/read:0"`
Pdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weightsЅ  "Ѕ2vdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ю
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"(
dnn/hiddenlayer_0/kernel  "(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
ж
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias "(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08

,dnn/hiddenlayer_0/batchnorm_0/gamma/part_0:01dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Assign@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Read/ReadVariableOp:0".
#dnn/hiddenlayer_0/batchnorm_0/gamma "(2=dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Initializer/ones:08

+dnn/hiddenlayer_0/batchnorm_0/beta/part_0:00dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Assign?dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Read/ReadVariableOp:0"-
"dnn/hiddenlayer_0/batchnorm_0/beta "(2=dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Initializer/zeros:08
ь
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel  "(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
ж
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias "(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08

,dnn/hiddenlayer_1/batchnorm_1/gamma/part_0:01dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Assign@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Read/ReadVariableOp:0".
#dnn/hiddenlayer_1/batchnorm_1/gamma "(2=dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Initializer/ones:08

+dnn/hiddenlayer_1/batchnorm_1/beta/part_0:00dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Assign?dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Read/ReadVariableOp:0"-
"dnn/hiddenlayer_1/batchnorm_1/beta "(2=dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Initializer/zeros:08
Щ
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel  "(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
Г
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_10/weights/part_0:04linear/linear_model/feature_10/weights/part_0/AssignClinear/linear_model/feature_10/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_10/weights  "(2Alinear/linear_model/feature_10/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_11/weights/part_0:04linear/linear_model/feature_11/weights/part_0/AssignClinear/linear_model/feature_11/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_11/weights  "(2Alinear/linear_model/feature_11/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_12/weights/part_0:04linear/linear_model/feature_12/weights/part_0/AssignClinear/linear_model/feature_12/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_12/weights  "(2Alinear/linear_model/feature_12/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_13/weights/part_0:04linear/linear_model/feature_13/weights/part_0/AssignClinear/linear_model/feature_13/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_13/weights  "(2Alinear/linear_model/feature_13/weights/part_0/Initializer/zeros:08

Clinear/linear_model/feature_14_embedding/embedding_weights/part_0:0Hlinear/linear_model/feature_14_embedding/embedding_weights/part_0/AssignHlinear/linear_model/feature_14_embedding/embedding_weights/part_0/read:0"H
:linear/linear_model/feature_14_embedding/embedding_weightsm  "m2`linear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
л
9linear/linear_model/feature_14_embedding/weights/part_0:0>linear/linear_model/feature_14_embedding/weights/part_0/AssignMlinear/linear_model/feature_14_embedding/weights/part_0/Read/ReadVariableOp:0">
0linear/linear_model/feature_14_embedding/weights  "(2Klinear/linear_model/feature_14_embedding/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_15/weights/part_0:04linear/linear_model/feature_15/weights/part_0/AssignClinear/linear_model/feature_15/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_15/weights  "(2Alinear/linear_model/feature_15/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_16/weights/part_0:04linear/linear_model/feature_16/weights/part_0/AssignClinear/linear_model/feature_16/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_16/weights  "(2Alinear/linear_model/feature_16/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_17/weights/part_0:04linear/linear_model/feature_17/weights/part_0/AssignClinear/linear_model/feature_17/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_17/weights  "(2Alinear/linear_model/feature_17/weights/part_0/Initializer/zeros:08

Clinear/linear_model/feature_18_embedding/embedding_weights/part_0:0Hlinear/linear_model/feature_18_embedding/embedding_weights/part_0/AssignHlinear/linear_model/feature_18_embedding/embedding_weights/part_0/read:0"H
:linear/linear_model/feature_18_embedding/embedding_weights   " 2`linear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
л
9linear/linear_model/feature_18_embedding/weights/part_0:0>linear/linear_model/feature_18_embedding/weights/part_0/AssignMlinear/linear_model/feature_18_embedding/weights/part_0/Read/ReadVariableOp:0">
0linear/linear_model/feature_18_embedding/weights  "(2Klinear/linear_model/feature_18_embedding/weights/part_0/Initializer/zeros:08

Blinear/linear_model/feature_2_embedding/embedding_weights/part_0:0Glinear/linear_model/feature_2_embedding/embedding_weights/part_0/AssignGlinear/linear_model/feature_2_embedding/embedding_weights/part_0/read:0"I
9linear/linear_model/feature_2_embedding/embedding_weights  "2_linear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ж
8linear/linear_model/feature_2_embedding/weights/part_0:0=linear/linear_model/feature_2_embedding/weights/part_0/AssignLlinear/linear_model/feature_2_embedding/weights/part_0/Read/ReadVariableOp:0"=
/linear/linear_model/feature_2_embedding/weights  "(2Jlinear/linear_model/feature_2_embedding/weights/part_0/Initializer/zeros:08
Є
.linear/linear_model/feature_3/weights/part_0:03linear/linear_model/feature_3/weights/part_0/AssignBlinear/linear_model/feature_3/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/feature_3/weights  "(2@linear/linear_model/feature_3/weights/part_0/Initializer/zeros:08
Є
.linear/linear_model/feature_4/weights/part_0:03linear/linear_model/feature_4/weights/part_0/AssignBlinear/linear_model/feature_4/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/feature_4/weights  "(2@linear/linear_model/feature_4/weights/part_0/Initializer/zeros:08
Є
.linear/linear_model/feature_5/weights/part_0:03linear/linear_model/feature_5/weights/part_0/AssignBlinear/linear_model/feature_5/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/feature_5/weights  "(2@linear/linear_model/feature_5/weights/part_0/Initializer/zeros:08
Є
.linear/linear_model/feature_6/weights/part_0:03linear/linear_model/feature_6/weights/part_0/AssignBlinear/linear_model/feature_6/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/feature_6/weights  "(2@linear/linear_model/feature_6/weights/part_0/Initializer/zeros:08

Blinear/linear_model/feature_7_embedding/embedding_weights/part_0:0Glinear/linear_model/feature_7_embedding/embedding_weights/part_0/AssignGlinear/linear_model/feature_7_embedding/embedding_weights/part_0/read:0"G
9linear/linear_model/feature_7_embedding/embedding_weights  "2_linear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ж
8linear/linear_model/feature_7_embedding/weights/part_0:0=linear/linear_model/feature_7_embedding/weights/part_0/AssignLlinear/linear_model/feature_7_embedding/weights/part_0/Read/ReadVariableOp:0"=
/linear/linear_model/feature_7_embedding/weights  "(2Jlinear/linear_model/feature_7_embedding/weights/part_0/Initializer/zeros:08

Blinear/linear_model/feature_8_embedding/embedding_weights/part_0:0Glinear/linear_model/feature_8_embedding/embedding_weights/part_0/AssignGlinear/linear_model/feature_8_embedding/embedding_weights/part_0/read:0"G
9linear/linear_model/feature_8_embedding/embedding_weights  "2_linear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ж
8linear/linear_model/feature_8_embedding/weights/part_0:0=linear/linear_model/feature_8_embedding/weights/part_0/AssignLlinear/linear_model/feature_8_embedding/weights/part_0/Read/ReadVariableOp:0"=
/linear/linear_model/feature_8_embedding/weights  "(2Jlinear/linear_model/feature_8_embedding/weights/part_0/Initializer/zeros:08

Blinear/linear_model/feature_9_embedding/embedding_weights/part_0:0Glinear/linear_model/feature_9_embedding/embedding_weights/part_0/AssignGlinear/linear_model/feature_9_embedding/embedding_weights/part_0/read:0"I
9linear/linear_model/feature_9_embedding/embedding_weightsЅ  "Ѕ2_linear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ж
8linear/linear_model/feature_9_embedding/weights/part_0:0=linear/linear_model/feature_9_embedding/weights/part_0/AssignLlinear/linear_model/feature_9_embedding/weights/part_0/Read/ReadVariableOp:0"=
/linear/linear_model/feature_9_embedding/weights  "(2Jlinear/linear_model/feature_9_embedding/weights/part_0/Initializer/zeros:08

)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08"љ
	summariesы
ш
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0
linear/bias:0
!linear/fraction_of_zero_weights:0
'linear/linear/fraction_of_zero_values:0
linear/linear/activation:0"q
	variablesq§p
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
њ
Zdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weightsm  "m2wdnn/input_from_feature_columns/input_layer/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
њ
Zdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights   " 2wdnn/input_from_feature_columns/input_layer/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ї
Ydnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/read:0"`
Pdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights  "2vdnn/input_from_feature_columns/input_layer/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ѕ
Ydnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/read:0"^
Pdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights  "2vdnn/input_from_feature_columns/input_layer/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ѕ
Ydnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/read:0"^
Pdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights  "2vdnn/input_from_feature_columns/input_layer/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ї
Ydnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/read:0"`
Pdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weightsЅ  "Ѕ2vdnn/input_from_feature_columns/input_layer/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ю
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"(
dnn/hiddenlayer_0/kernel  "(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
ж
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias "(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08

,dnn/hiddenlayer_0/batchnorm_0/gamma/part_0:01dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Assign@dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Read/ReadVariableOp:0".
#dnn/hiddenlayer_0/batchnorm_0/gamma "(2=dnn/hiddenlayer_0/batchnorm_0/gamma/part_0/Initializer/ones:08

+dnn/hiddenlayer_0/batchnorm_0/beta/part_0:00dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Assign?dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Read/ReadVariableOp:0"-
"dnn/hiddenlayer_0/batchnorm_0/beta "(2=dnn/hiddenlayer_0/batchnorm_0/beta/part_0/Initializer/zeros:08
с
+dnn/hiddenlayer_0/batchnorm_0/moving_mean:00dnn/hiddenlayer_0/batchnorm_0/moving_mean/Assign?dnn/hiddenlayer_0/batchnorm_0/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_0/batchnorm_0/moving_mean/Initializer/zeros:0
№
/dnn/hiddenlayer_0/batchnorm_0/moving_variance:04dnn/hiddenlayer_0/batchnorm_0/moving_variance/AssignCdnn/hiddenlayer_0/batchnorm_0/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_0/batchnorm_0/moving_variance/Initializer/ones:0
ь
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel  "(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
ж
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias "(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08

,dnn/hiddenlayer_1/batchnorm_1/gamma/part_0:01dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Assign@dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Read/ReadVariableOp:0".
#dnn/hiddenlayer_1/batchnorm_1/gamma "(2=dnn/hiddenlayer_1/batchnorm_1/gamma/part_0/Initializer/ones:08

+dnn/hiddenlayer_1/batchnorm_1/beta/part_0:00dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Assign?dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Read/ReadVariableOp:0"-
"dnn/hiddenlayer_1/batchnorm_1/beta "(2=dnn/hiddenlayer_1/batchnorm_1/beta/part_0/Initializer/zeros:08
с
+dnn/hiddenlayer_1/batchnorm_1/moving_mean:00dnn/hiddenlayer_1/batchnorm_1/moving_mean/Assign?dnn/hiddenlayer_1/batchnorm_1/moving_mean/Read/ReadVariableOp:0(2=dnn/hiddenlayer_1/batchnorm_1/moving_mean/Initializer/zeros:0
№
/dnn/hiddenlayer_1/batchnorm_1/moving_variance:04dnn/hiddenlayer_1/batchnorm_1/moving_variance/AssignCdnn/hiddenlayer_1/batchnorm_1/moving_variance/Read/ReadVariableOp:0(2@dnn/hiddenlayer_1/batchnorm_1/moving_variance/Initializer/ones:0
Щ
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel  "(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
Г
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_10/weights/part_0:04linear/linear_model/feature_10/weights/part_0/AssignClinear/linear_model/feature_10/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_10/weights  "(2Alinear/linear_model/feature_10/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_11/weights/part_0:04linear/linear_model/feature_11/weights/part_0/AssignClinear/linear_model/feature_11/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_11/weights  "(2Alinear/linear_model/feature_11/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_12/weights/part_0:04linear/linear_model/feature_12/weights/part_0/AssignClinear/linear_model/feature_12/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_12/weights  "(2Alinear/linear_model/feature_12/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_13/weights/part_0:04linear/linear_model/feature_13/weights/part_0/AssignClinear/linear_model/feature_13/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_13/weights  "(2Alinear/linear_model/feature_13/weights/part_0/Initializer/zeros:08

Clinear/linear_model/feature_14_embedding/embedding_weights/part_0:0Hlinear/linear_model/feature_14_embedding/embedding_weights/part_0/AssignHlinear/linear_model/feature_14_embedding/embedding_weights/part_0/read:0"H
:linear/linear_model/feature_14_embedding/embedding_weightsm  "m2`linear/linear_model/feature_14_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
л
9linear/linear_model/feature_14_embedding/weights/part_0:0>linear/linear_model/feature_14_embedding/weights/part_0/AssignMlinear/linear_model/feature_14_embedding/weights/part_0/Read/ReadVariableOp:0">
0linear/linear_model/feature_14_embedding/weights  "(2Klinear/linear_model/feature_14_embedding/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_15/weights/part_0:04linear/linear_model/feature_15/weights/part_0/AssignClinear/linear_model/feature_15/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_15/weights  "(2Alinear/linear_model/feature_15/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_16/weights/part_0:04linear/linear_model/feature_16/weights/part_0/AssignClinear/linear_model/feature_16/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_16/weights  "(2Alinear/linear_model/feature_16/weights/part_0/Initializer/zeros:08
Љ
/linear/linear_model/feature_17/weights/part_0:04linear/linear_model/feature_17/weights/part_0/AssignClinear/linear_model/feature_17/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/feature_17/weights  "(2Alinear/linear_model/feature_17/weights/part_0/Initializer/zeros:08

Clinear/linear_model/feature_18_embedding/embedding_weights/part_0:0Hlinear/linear_model/feature_18_embedding/embedding_weights/part_0/AssignHlinear/linear_model/feature_18_embedding/embedding_weights/part_0/read:0"H
:linear/linear_model/feature_18_embedding/embedding_weights   " 2`linear/linear_model/feature_18_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
л
9linear/linear_model/feature_18_embedding/weights/part_0:0>linear/linear_model/feature_18_embedding/weights/part_0/AssignMlinear/linear_model/feature_18_embedding/weights/part_0/Read/ReadVariableOp:0">
0linear/linear_model/feature_18_embedding/weights  "(2Klinear/linear_model/feature_18_embedding/weights/part_0/Initializer/zeros:08

Blinear/linear_model/feature_2_embedding/embedding_weights/part_0:0Glinear/linear_model/feature_2_embedding/embedding_weights/part_0/AssignGlinear/linear_model/feature_2_embedding/embedding_weights/part_0/read:0"I
9linear/linear_model/feature_2_embedding/embedding_weights  "2_linear/linear_model/feature_2_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ж
8linear/linear_model/feature_2_embedding/weights/part_0:0=linear/linear_model/feature_2_embedding/weights/part_0/AssignLlinear/linear_model/feature_2_embedding/weights/part_0/Read/ReadVariableOp:0"=
/linear/linear_model/feature_2_embedding/weights  "(2Jlinear/linear_model/feature_2_embedding/weights/part_0/Initializer/zeros:08
Є
.linear/linear_model/feature_3/weights/part_0:03linear/linear_model/feature_3/weights/part_0/AssignBlinear/linear_model/feature_3/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/feature_3/weights  "(2@linear/linear_model/feature_3/weights/part_0/Initializer/zeros:08
Є
.linear/linear_model/feature_4/weights/part_0:03linear/linear_model/feature_4/weights/part_0/AssignBlinear/linear_model/feature_4/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/feature_4/weights  "(2@linear/linear_model/feature_4/weights/part_0/Initializer/zeros:08
Є
.linear/linear_model/feature_5/weights/part_0:03linear/linear_model/feature_5/weights/part_0/AssignBlinear/linear_model/feature_5/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/feature_5/weights  "(2@linear/linear_model/feature_5/weights/part_0/Initializer/zeros:08
Є
.linear/linear_model/feature_6/weights/part_0:03linear/linear_model/feature_6/weights/part_0/AssignBlinear/linear_model/feature_6/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/feature_6/weights  "(2@linear/linear_model/feature_6/weights/part_0/Initializer/zeros:08

Blinear/linear_model/feature_7_embedding/embedding_weights/part_0:0Glinear/linear_model/feature_7_embedding/embedding_weights/part_0/AssignGlinear/linear_model/feature_7_embedding/embedding_weights/part_0/read:0"G
9linear/linear_model/feature_7_embedding/embedding_weights  "2_linear/linear_model/feature_7_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ж
8linear/linear_model/feature_7_embedding/weights/part_0:0=linear/linear_model/feature_7_embedding/weights/part_0/AssignLlinear/linear_model/feature_7_embedding/weights/part_0/Read/ReadVariableOp:0"=
/linear/linear_model/feature_7_embedding/weights  "(2Jlinear/linear_model/feature_7_embedding/weights/part_0/Initializer/zeros:08

Blinear/linear_model/feature_8_embedding/embedding_weights/part_0:0Glinear/linear_model/feature_8_embedding/embedding_weights/part_0/AssignGlinear/linear_model/feature_8_embedding/embedding_weights/part_0/read:0"G
9linear/linear_model/feature_8_embedding/embedding_weights  "2_linear/linear_model/feature_8_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ж
8linear/linear_model/feature_8_embedding/weights/part_0:0=linear/linear_model/feature_8_embedding/weights/part_0/AssignLlinear/linear_model/feature_8_embedding/weights/part_0/Read/ReadVariableOp:0"=
/linear/linear_model/feature_8_embedding/weights  "(2Jlinear/linear_model/feature_8_embedding/weights/part_0/Initializer/zeros:08

Blinear/linear_model/feature_9_embedding/embedding_weights/part_0:0Glinear/linear_model/feature_9_embedding/embedding_weights/part_0/AssignGlinear/linear_model/feature_9_embedding/embedding_weights/part_0/read:0"I
9linear/linear_model/feature_9_embedding/embedding_weightsЅ  "Ѕ2_linear/linear_model/feature_9_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
ж
8linear/linear_model/feature_9_embedding/weights/part_0:0=linear/linear_model/feature_9_embedding/weights/part_0/AssignLlinear/linear_model/feature_9_embedding/weights/part_0/Read/ReadVariableOp:0"=
/linear/linear_model/feature_9_embedding/weights  "(2Jlinear/linear_model/feature_9_embedding/weights/part_0/Initializer/zeros:08

)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08*
predict
5
examples)
input_example_tensor:0џџџџџџџџџ+
predictions
add:0џџџџџџџџџtensorflow/serving/predict*

regressionz
3
inputs)
input_example_tensor:0џџџџџџџџџ'
outputs
add:0џџџџџџџџџtensorflow/serving/regress*
serving_defaultz
3
inputs)
input_example_tensor:0џџџџџџџџџ'
outputs
add:0џџџџџџџџџtensorflow/serving/regress