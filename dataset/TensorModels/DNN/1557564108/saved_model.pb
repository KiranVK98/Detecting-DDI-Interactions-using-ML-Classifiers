??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
?
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
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
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
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
?
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
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?
9
VarIsInitializedOp
resource
is_initialized
?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.13.12
b'unknown'ϛ

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
?
global_step
VariableV2*
_class
loc:@global_step*
	container *
shape: *
dtype0	*
_output_shapes
: *
shared_name 
?
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
Y
input_example_tensorPlaceholder*
dtype0*
_output_shapes
:*
shape:
U
ParseExample/ConstConst*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_1Const*
dtype0*
_output_shapes
: *
valueB 
W
ParseExample/Const_2Const*
valueB	 *
dtype0	*
_output_shapes
: 
W
ParseExample/Const_3Const*
valueB	 *
dtype0	*
_output_shapes
: 
W
ParseExample/Const_4Const*
dtype0	*
_output_shapes
: *
valueB	 
W
ParseExample/Const_5Const*
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
p
&ParseExample/ParseExample/dense_keys_0Const*
valueB B	drug1_prr*
dtype0*
_output_shapes
: 
p
&ParseExample/ParseExample/dense_keys_1Const*
dtype0*
_output_shapes
: *
valueB B	drug2_prr
m
&ParseExample/ParseExample/dense_keys_2Const*
valueB Bdrug_1*
dtype0*
_output_shapes
: 
m
&ParseExample/ParseExample/dense_keys_3Const*
dtype0*
_output_shapes
: *
valueB Bdrug_2
l
&ParseExample/ParseExample/dense_keys_4Const*
valueB Bevent*
dtype0*
_output_shapes
: 
j
&ParseExample/ParseExample/dense_keys_5Const*
valueB	 Bprr*
dtype0*
_output_shapes
: 
?
ParseExample/ParseExampleParseExampleinput_example_tensorParseExample/ParseExample/names&ParseExample/ParseExample/dense_keys_0&ParseExample/ParseExample/dense_keys_1&ParseExample/ParseExample/dense_keys_2&ParseExample/ParseExample/dense_keys_3&ParseExample/ParseExample/dense_keys_4&ParseExample/ParseExample/dense_keys_5ParseExample/ConstParseExample/Const_1ParseExample/Const_2ParseExample/Const_3ParseExample/Const_4ParseExample/Const_5*z
_output_shapesh
f:????????? :????????? :????????? :????????? :????????? :????????? *
Nsparse **
dense_shapes
: : : : : : *
sparse_types
 *
Tdense

2			*
Ndense
?
:dnn/input_from_feature_columns/input_layer/drug1_prr/ShapeShapeParseExample/ParseExample*
T0*
out_type0*
_output_shapes
:
?
Hdnn/input_from_feature_columns/input_layer/drug1_prr/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Jdnn/input_from_feature_columns/input_layer/drug1_prr/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Jdnn/input_from_feature_columns/input_layer/drug1_prr/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Bdnn/input_from_feature_columns/input_layer/drug1_prr/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/drug1_prr/ShapeHdnn/input_from_feature_columns/input_layer/drug1_prr/strided_slice/stackJdnn/input_from_feature_columns/input_layer/drug1_prr/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/drug1_prr/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
?
Ddnn/input_from_feature_columns/input_layer/drug1_prr/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
?
Bdnn/input_from_feature_columns/input_layer/drug1_prr/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/drug1_prr/strided_sliceDdnn/input_from_feature_columns/input_layer/drug1_prr/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
?
<dnn/input_from_feature_columns/input_layer/drug1_prr/ReshapeReshapeParseExample/ParseExampleBdnn/input_from_feature_columns/input_layer/drug1_prr/Reshape/shape*
T0*
Tshape0*
_output_shapes

: 
?
:dnn/input_from_feature_columns/input_layer/drug2_prr/ShapeShapeParseExample/ParseExample:1*
T0*
out_type0*
_output_shapes
:
?
Hdnn/input_from_feature_columns/input_layer/drug2_prr/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Jdnn/input_from_feature_columns/input_layer/drug2_prr/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Jdnn/input_from_feature_columns/input_layer/drug2_prr/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Bdnn/input_from_feature_columns/input_layer/drug2_prr/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/drug2_prr/ShapeHdnn/input_from_feature_columns/input_layer/drug2_prr/strided_slice/stackJdnn/input_from_feature_columns/input_layer/drug2_prr/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/drug2_prr/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
Ddnn/input_from_feature_columns/input_layer/drug2_prr/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Bdnn/input_from_feature_columns/input_layer/drug2_prr/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/drug2_prr/strided_sliceDdnn/input_from_feature_columns/input_layer/drug2_prr/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
?
<dnn/input_from_feature_columns/input_layer/drug2_prr/ReshapeReshapeParseExample/ParseExample:1Bdnn/input_from_feature_columns/input_layer/drug2_prr/Reshape/shape*
T0*
Tshape0*
_output_shapes

: 
?
9dnn/input_from_feature_columns/input_layer/drug_1/ToFloatCastParseExample/ParseExample:2*

SrcT0	*
Truncate( *%
_output_shapes
:????????? *

DstT0
?
7dnn/input_from_feature_columns/input_layer/drug_1/ShapeShape9dnn/input_from_feature_columns/input_layer/drug_1/ToFloat*
_output_shapes
:*
T0*
out_type0
?
Ednn/input_from_feature_columns/input_layer/drug_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Gdnn/input_from_feature_columns/input_layer/drug_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Gdnn/input_from_feature_columns/input_layer/drug_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/drug_1/strided_sliceStridedSlice7dnn/input_from_feature_columns/input_layer/drug_1/ShapeEdnn/input_from_feature_columns/input_layer/drug_1/strided_slice/stackGdnn/input_from_feature_columns/input_layer/drug_1/strided_slice/stack_1Gdnn/input_from_feature_columns/input_layer/drug_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
Adnn/input_from_feature_columns/input_layer/drug_1/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
?dnn/input_from_feature_columns/input_layer/drug_1/Reshape/shapePack?dnn/input_from_feature_columns/input_layer/drug_1/strided_sliceAdnn/input_from_feature_columns/input_layer/drug_1/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
?
9dnn/input_from_feature_columns/input_layer/drug_1/ReshapeReshape9dnn/input_from_feature_columns/input_layer/drug_1/ToFloat?dnn/input_from_feature_columns/input_layer/drug_1/Reshape/shape*
_output_shapes

: *
T0*
Tshape0
?
9dnn/input_from_feature_columns/input_layer/drug_2/ToFloatCastParseExample/ParseExample:3*

SrcT0	*
Truncate( *%
_output_shapes
:????????? *

DstT0
?
7dnn/input_from_feature_columns/input_layer/drug_2/ShapeShape9dnn/input_from_feature_columns/input_layer/drug_2/ToFloat*
T0*
out_type0*
_output_shapes
:
?
Ednn/input_from_feature_columns/input_layer/drug_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Gdnn/input_from_feature_columns/input_layer/drug_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Gdnn/input_from_feature_columns/input_layer/drug_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/drug_2/strided_sliceStridedSlice7dnn/input_from_feature_columns/input_layer/drug_2/ShapeEdnn/input_from_feature_columns/input_layer/drug_2/strided_slice/stackGdnn/input_from_feature_columns/input_layer/drug_2/strided_slice/stack_1Gdnn/input_from_feature_columns/input_layer/drug_2/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
Adnn/input_from_feature_columns/input_layer/drug_2/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
?dnn/input_from_feature_columns/input_layer/drug_2/Reshape/shapePack?dnn/input_from_feature_columns/input_layer/drug_2/strided_sliceAdnn/input_from_feature_columns/input_layer/drug_2/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
?
9dnn/input_from_feature_columns/input_layer/drug_2/ReshapeReshape9dnn/input_from_feature_columns/input_layer/drug_2/ToFloat?dnn/input_from_feature_columns/input_layer/drug_2/Reshape/shape*
T0*
Tshape0*
_output_shapes

: 
?
8dnn/input_from_feature_columns/input_layer/event/ToFloatCastParseExample/ParseExample:4*

SrcT0	*
Truncate( *%
_output_shapes
:????????? *

DstT0
?
6dnn/input_from_feature_columns/input_layer/event/ShapeShape8dnn/input_from_feature_columns/input_layer/event/ToFloat*
_output_shapes
:*
T0*
out_type0
?
Ddnn/input_from_feature_columns/input_layer/event/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Fdnn/input_from_feature_columns/input_layer/event/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Fdnn/input_from_feature_columns/input_layer/event/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
>dnn/input_from_feature_columns/input_layer/event/strided_sliceStridedSlice6dnn/input_from_feature_columns/input_layer/event/ShapeDdnn/input_from_feature_columns/input_layer/event/strided_slice/stackFdnn/input_from_feature_columns/input_layer/event/strided_slice/stack_1Fdnn/input_from_feature_columns/input_layer/event/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
@dnn/input_from_feature_columns/input_layer/event/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
>dnn/input_from_feature_columns/input_layer/event/Reshape/shapePack>dnn/input_from_feature_columns/input_layer/event/strided_slice@dnn/input_from_feature_columns/input_layer/event/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
?
8dnn/input_from_feature_columns/input_layer/event/ReshapeReshape8dnn/input_from_feature_columns/input_layer/event/ToFloat>dnn/input_from_feature_columns/input_layer/event/Reshape/shape*
T0*
Tshape0*
_output_shapes

: 
?
4dnn/input_from_feature_columns/input_layer/prr/ShapeShapeParseExample/ParseExample:5*
T0*
out_type0*
_output_shapes
:
?
Bdnn/input_from_feature_columns/input_layer/prr/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Ddnn/input_from_feature_columns/input_layer/prr/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Ddnn/input_from_feature_columns/input_layer/prr/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
<dnn/input_from_feature_columns/input_layer/prr/strided_sliceStridedSlice4dnn/input_from_feature_columns/input_layer/prr/ShapeBdnn/input_from_feature_columns/input_layer/prr/strided_slice/stackDdnn/input_from_feature_columns/input_layer/prr/strided_slice/stack_1Ddnn/input_from_feature_columns/input_layer/prr/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
>dnn/input_from_feature_columns/input_layer/prr/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
<dnn/input_from_feature_columns/input_layer/prr/Reshape/shapePack<dnn/input_from_feature_columns/input_layer/prr/strided_slice>dnn/input_from_feature_columns/input_layer/prr/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
?
6dnn/input_from_feature_columns/input_layer/prr/ReshapeReshapeParseExample/ParseExample:5<dnn/input_from_feature_columns/input_layer/prr/Reshape/shape*
T0*
Tshape0*
_output_shapes

: 
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
?
1dnn/input_from_feature_columns/input_layer/concatConcatV2<dnn/input_from_feature_columns/input_layer/drug1_prr/Reshape<dnn/input_from_feature_columns/input_layer/drug2_prr/Reshape9dnn/input_from_feature_columns/input_layer/drug_1/Reshape9dnn/input_from_feature_columns/input_layer/drug_2/Reshape8dnn/input_from_feature_columns/input_layer/event/Reshape6dnn/input_from_feature_columns/input_layer/prr/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
T0*
N*
_output_shapes

: *

Tidx0
?
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"   
   *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *q??*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *q??*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
?
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:
*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:

?
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:

?
dnn/hiddenlayer_0/kernel/part_0VarHandleOp*0
shared_name!dnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container *
shape
:
*
dtype0*
_output_shapes
: 
?
@dnn/hiddenlayer_0/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_0/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
?
3dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:
*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
?
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*
valueB
*    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:

?
dnn/hiddenlayer_0/bias/part_0VarHandleOp*
	container *
shape:
*
dtype0*
_output_shapes
: *.
shared_namednn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
?
>dnn/hiddenlayer_0/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_0/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0
?
1dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:

?
'dnn/hiddenlayer_0/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:

v
dnn/hiddenlayer_0/kernelIdentity'dnn/hiddenlayer_0/kernel/ReadVariableOp*
T0*
_output_shapes

:

?
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
_output_shapes

: 
*
transpose_a( *
transpose_b( *
T0

%dnn/hiddenlayer_0/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:

n
dnn/hiddenlayer_0/biasIdentity%dnn/hiddenlayer_0/bias/ReadVariableOp*
T0*
_output_shapes
:

?
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*
data_formatNHWC*
_output_shapes

: 

`
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*
_output_shapes

: 

X
dnn/zero_fraction/SizeConst*
value	B	 R *
dtype0	*
_output_shapes
: 
c
dnn/zero_fraction/LessEqual/yConst*
dtype0	*
_output_shapes
: *
valueB	 R????
?
dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
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
?
*dnn/zero_fraction/cond/count_nonzero/zerosConst ^dnn/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
?
-dnn/zero_fraction/cond/count_nonzero/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1*dnn/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

: 

?
4dnn/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*$
_output_shapes
: 
: 
*
T0*)
_class
loc:@dnn/hiddenlayer_0/Relu
?
)dnn/zero_fraction/cond/count_nonzero/CastCast-dnn/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

: 
*

DstT0
?
*dnn/zero_fraction/cond/count_nonzero/ConstConst ^dnn/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
2dnn/zero_fraction/cond/count_nonzero/nonzero_countSum)dnn/zero_fraction/cond/count_nonzero/Cast*dnn/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
?
dnn/zero_fraction/cond/CastCast2dnn/zero_fraction/cond/count_nonzero/nonzero_count*
Truncate( *
_output_shapes
: *

DstT0	*

SrcT0
?
,dnn/zero_fraction/cond/count_nonzero_1/zerosConst ^dnn/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
?
/dnn/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch,dnn/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

: 

?
6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_0/Relu*$
_output_shapes
: 
: 

?
+dnn/zero_fraction/cond/count_nonzero_1/CastCast/dnn/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

: 
*

DstT0	
?
,dnn/zero_fraction/cond/count_nonzero_1/ConstConst ^dnn/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countSum+dnn/zero_fraction/cond/count_nonzero_1/Cast,dnn/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0	
?
dnn/zero_fraction/cond/MergeMerge4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countdnn/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
?
)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
T0*
_output_shapes
: 
?
$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 
?
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: *
T0
?
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
   
   *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
:
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *?7?*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *?7?*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
?
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:

*

seed 
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes

:

*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
?
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:

*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
?
dnn/hiddenlayer_1/kernel/part_0VarHandleOp*0
shared_name!dnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container *
shape
:

*
dtype0*
_output_shapes
: 
?
@dnn/hiddenlayer_1/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_1/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0
?
3dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:


?
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
valueB
*    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
?
dnn/hiddenlayer_1/bias/part_0VarHandleOp*.
shared_namednn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container *
shape:
*
dtype0*
_output_shapes
: 
?
>dnn/hiddenlayer_1/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_1/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*
dtype0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
?
1dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
?
'dnn/hiddenlayer_1/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:


v
dnn/hiddenlayer_1/kernelIdentity'dnn/hiddenlayer_1/kernel/ReadVariableOp*
_output_shapes

:

*
T0
?
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
_output_shapes

: 
*
transpose_a( *
transpose_b( *
T0

%dnn/hiddenlayer_1/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:

n
dnn/hiddenlayer_1/biasIdentity%dnn/hiddenlayer_1/bias/ReadVariableOp*
T0*
_output_shapes
:

?
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*
data_formatNHWC*
_output_shapes

: 

`
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*
_output_shapes

: 

Z
dnn/zero_fraction_1/SizeConst*
dtype0	*
_output_shapes
: *
value	B	 R 
e
dnn/zero_fraction_1/LessEqual/yConst*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 
?
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
 dnn/zero_fraction_1/cond/pred_idIdentitydnn/zero_fraction_1/LessEqual*
_output_shapes
: *
T0

?
,dnn/zero_fraction_1/cond/count_nonzero/zerosConst"^dnn/zero_fraction_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
/dnn/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_1/cond/count_nonzero/zeros*
T0*
_output_shapes

: 

?
6dnn/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_1/Relu*$
_output_shapes
: 
: 

?
+dnn/zero_fraction_1/cond/count_nonzero/CastCast/dnn/zero_fraction_1/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

: 
*

DstT0
?
,dnn/zero_fraction_1/cond/count_nonzero/ConstConst"^dnn/zero_fraction_1/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
?
4dnn/zero_fraction_1/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_1/cond/count_nonzero/Cast,dnn/zero_fraction_1/cond/count_nonzero/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
?
dnn/zero_fraction_1/cond/CastCast4dnn/zero_fraction_1/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
?
.dnn/zero_fraction_1/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_1/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
?
1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_1/cond/count_nonzero_1/zeros*
T0*
_output_shapes

: 

?
8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_1/Relu*$
_output_shapes
: 
: 

?
-dnn/zero_fraction_1/cond/count_nonzero_1/CastCast1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

: 
*

DstT0	
?
.dnn/zero_fraction_1/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_1/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_1/cond/count_nonzero_1/Cast.dnn/zero_fraction_1/cond/count_nonzero_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0	
?
dnn/zero_fraction_1/cond/MergeMerge6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_1/cond/Cast*
N*
_output_shapes
: : *
T0	
?
*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Sizednn/zero_fraction_1/cond/Merge*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_1/counts_to_fraction/CastCast*dnn/zero_fraction_1/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
-dnn/zero_fraction_1/counts_to_fraction/Cast_1Castdnn/zero_fraction_1/Size*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
T0*
_output_shapes
: 
?
$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
?
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: *
T0
?
@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
   
   *2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
:
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?7?*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *?7?*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 
?
Hdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:

*

seed 
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: 
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:


?
:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:


?
dnn/hiddenlayer_2/kernel/part_0VarHandleOp*0
shared_name!dnn/hiddenlayer_2/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
	container *
shape
:

*
dtype0*
_output_shapes
: 
?
@dnn/hiddenlayer_2/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_2/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_2/kernel/part_0:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0
?
3dnn/hiddenlayer_2/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes

:


?
/dnn/hiddenlayer_2/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
valueB
*    *0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0
?
dnn/hiddenlayer_2/bias/part_0VarHandleOp*
dtype0*
_output_shapes
: *.
shared_namednn/hiddenlayer_2/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
	container *
shape:

?
>dnn/hiddenlayer_2/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_2/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_2/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_2/bias/part_0/dnn/hiddenlayer_2/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
dtype0
?
1dnn/hiddenlayer_2/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
:
*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0
?
'dnn/hiddenlayer_2/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes

:


v
dnn/hiddenlayer_2/kernelIdentity'dnn/hiddenlayer_2/kernel/ReadVariableOp*
T0*
_output_shapes

:


?
dnn/hiddenlayer_2/MatMulMatMuldnn/hiddenlayer_1/Reludnn/hiddenlayer_2/kernel*
T0*
_output_shapes

: 
*
transpose_a( *
transpose_b( 

%dnn/hiddenlayer_2/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
:

n
dnn/hiddenlayer_2/biasIdentity%dnn/hiddenlayer_2/bias/ReadVariableOp*
T0*
_output_shapes
:

?
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/bias*
T0*
data_formatNHWC*
_output_shapes

: 

`
dnn/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*
_output_shapes

: 

Z
dnn/zero_fraction_2/SizeConst*
dtype0	*
_output_shapes
: *
value	B	 R 
e
dnn/zero_fraction_2/LessEqual/yConst*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
dnn/zero_fraction_2/LessEqual	LessEqualdnn/zero_fraction_2/Sizednn/zero_fraction_2/LessEqual/y*
T0	*
_output_shapes
: 
?
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
?
,dnn/zero_fraction_2/cond/count_nonzero/zerosConst"^dnn/zero_fraction_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
/dnn/zero_fraction_2/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_2/cond/count_nonzero/zeros*
T0*
_output_shapes

: 

?
6dnn/zero_fraction_2/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_2/Relu dnn/zero_fraction_2/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_2/Relu*$
_output_shapes
: 
: 

?
+dnn/zero_fraction_2/cond/count_nonzero/CastCast/dnn/zero_fraction_2/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

: 
*

DstT0
?
,dnn/zero_fraction_2/cond/count_nonzero/ConstConst"^dnn/zero_fraction_2/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
4dnn/zero_fraction_2/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_2/cond/count_nonzero/Cast,dnn/zero_fraction_2/cond/count_nonzero/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
?
dnn/zero_fraction_2/cond/CastCast4dnn/zero_fraction_2/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
?
.dnn/zero_fraction_2/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_2/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
?
1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_2/cond/count_nonzero_1/zeros*
_output_shapes

: 
*
T0
?
8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_2/Relu dnn/zero_fraction_2/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_2/Relu*$
_output_shapes
: 
: 

?
-dnn/zero_fraction_2/cond/count_nonzero_1/CastCast1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

: 
*

DstT0	
?
.dnn/zero_fraction_2/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_2/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_2/cond/count_nonzero_1/Cast.dnn/zero_fraction_2/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: *
	keep_dims( *

Tidx0
?
dnn/zero_fraction_2/cond/MergeMerge6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_2/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
*dnn/zero_fraction_2/counts_to_fraction/subSubdnn/zero_fraction_2/Sizednn/zero_fraction_2/cond/Merge*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_2/counts_to_fraction/CastCast*dnn/zero_fraction_2/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
-dnn/zero_fraction_2/counts_to_fraction/Cast_1Castdnn/zero_fraction_2/Size*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
?
.dnn/zero_fraction_2/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_2/counts_to_fraction/Cast-dnn/zero_fraction_2/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_2/fractionIdentity.dnn/zero_fraction_2/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_2/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
-dnn/dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/fraction*
T0*
_output_shapes
: 
?
$dnn/dnn/hiddenlayer_2/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_2/activation*
dtype0*
_output_shapes
: 
?
 dnn/dnn/hiddenlayer_2/activationHistogramSummary$dnn/dnn/hiddenlayer_2/activation/tagdnn/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
?
@dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
   
   *2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes
:
?
>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *?7?*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes
: 
?
>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *?7?*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes
: 
?
Hdnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:

*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
seed2 
?
>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
_output_shapes
: 
?
>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
_output_shapes

:


?
:dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
_output_shapes

:


?
dnn/hiddenlayer_3/kernel/part_0VarHandleOp*
	container *
shape
:

*
dtype0*
_output_shapes
: *0
shared_name!dnn/hiddenlayer_3/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0
?
@dnn/hiddenlayer_3/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_3/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_3/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_3/kernel/part_0:dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0
?
3dnn/hiddenlayer_3/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes

:


?
/dnn/hiddenlayer_3/bias/part_0/Initializer/zerosConst*
valueB
*    *0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0*
dtype0*
_output_shapes
:

?
dnn/hiddenlayer_3/bias/part_0VarHandleOp*
dtype0*
_output_shapes
: *.
shared_namednn/hiddenlayer_3/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0*
	container *
shape:

?
>dnn/hiddenlayer_3/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_3/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_3/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_3/bias/part_0/dnn/hiddenlayer_3/bias/part_0/Initializer/zeros*
dtype0*0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0
?
1dnn/hiddenlayer_3/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0*
dtype0*
_output_shapes
:

?
'dnn/hiddenlayer_3/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes

:


v
dnn/hiddenlayer_3/kernelIdentity'dnn/hiddenlayer_3/kernel/ReadVariableOp*
T0*
_output_shapes

:


?
dnn/hiddenlayer_3/MatMulMatMuldnn/hiddenlayer_2/Reludnn/hiddenlayer_3/kernel*
T0*
_output_shapes

: 
*
transpose_a( *
transpose_b( 

%dnn/hiddenlayer_3/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/bias/part_0*
dtype0*
_output_shapes
:

n
dnn/hiddenlayer_3/biasIdentity%dnn/hiddenlayer_3/bias/ReadVariableOp*
T0*
_output_shapes
:

?
dnn/hiddenlayer_3/BiasAddBiasAdddnn/hiddenlayer_3/MatMuldnn/hiddenlayer_3/bias*
T0*
data_formatNHWC*
_output_shapes

: 

`
dnn/hiddenlayer_3/ReluReludnn/hiddenlayer_3/BiasAdd*
T0*
_output_shapes

: 

Z
dnn/zero_fraction_3/SizeConst*
value	B	 R *
dtype0	*
_output_shapes
: 
e
dnn/zero_fraction_3/LessEqual/yConst*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
dnn/zero_fraction_3/LessEqual	LessEqualdnn/zero_fraction_3/Sizednn/zero_fraction_3/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_3/cond/SwitchSwitchdnn/zero_fraction_3/LessEqualdnn/zero_fraction_3/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_3/cond/switch_tIdentity!dnn/zero_fraction_3/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_3/cond/switch_fIdentitydnn/zero_fraction_3/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_3/cond/pred_idIdentitydnn/zero_fraction_3/LessEqual*
T0
*
_output_shapes
: 
?
,dnn/zero_fraction_3/cond/count_nonzero/zerosConst"^dnn/zero_fraction_3/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
/dnn/zero_fraction_3/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_3/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_3/cond/count_nonzero/zeros*
_output_shapes

: 
*
T0
?
6dnn/zero_fraction_3/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_3/Relu dnn/zero_fraction_3/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_3/Relu*$
_output_shapes
: 
: 

?
+dnn/zero_fraction_3/cond/count_nonzero/CastCast/dnn/zero_fraction_3/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

: 
*

DstT0
?
,dnn/zero_fraction_3/cond/count_nonzero/ConstConst"^dnn/zero_fraction_3/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
4dnn/zero_fraction_3/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_3/cond/count_nonzero/Cast,dnn/zero_fraction_3/cond/count_nonzero/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
?
dnn/zero_fraction_3/cond/CastCast4dnn/zero_fraction_3/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
?
.dnn/zero_fraction_3/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_3/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
1dnn/zero_fraction_3/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_3/cond/count_nonzero_1/zeros*
T0*
_output_shapes

: 

?
8dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_3/Relu dnn/zero_fraction_3/cond/pred_id*$
_output_shapes
: 
: 
*
T0*)
_class
loc:@dnn/hiddenlayer_3/Relu
?
-dnn/zero_fraction_3/cond/count_nonzero_1/CastCast1dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

: 
*

DstT0	
?
.dnn/zero_fraction_3/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_3/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
6dnn/zero_fraction_3/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_3/cond/count_nonzero_1/Cast.dnn/zero_fraction_3/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: *
	keep_dims( *

Tidx0
?
dnn/zero_fraction_3/cond/MergeMerge6dnn/zero_fraction_3/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_3/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
*dnn/zero_fraction_3/counts_to_fraction/subSubdnn/zero_fraction_3/Sizednn/zero_fraction_3/cond/Merge*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_3/counts_to_fraction/CastCast*dnn/zero_fraction_3/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
-dnn/zero_fraction_3/counts_to_fraction/Cast_1Castdnn/zero_fraction_3/Size*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
.dnn/zero_fraction_3/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_3/counts_to_fraction/Cast-dnn/zero_fraction_3/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
y
dnn/zero_fraction_3/fractionIdentity.dnn/zero_fraction_3/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2dnn/dnn/hiddenlayer_3/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_3/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
-dnn/dnn/hiddenlayer_3/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_3/fraction_of_zero_values/tagsdnn/zero_fraction_3/fraction*
T0*
_output_shapes
: 
?
$dnn/dnn/hiddenlayer_3/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_3/activation*
dtype0*
_output_shapes
: 
?
 dnn/dnn/hiddenlayer_3/activationHistogramSummary$dnn/dnn/hiddenlayer_3/activation/tagdnn/hiddenlayer_3/Relu*
T0*
_output_shapes
: 
?
@dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
   
   *2
_class(
&$loc:@dnn/hiddenlayer_4/kernel/part_0*
dtype0*
_output_shapes
:
?
>dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *?7?*2
_class(
&$loc:@dnn/hiddenlayer_4/kernel/part_0*
dtype0*
_output_shapes
: 
?
>dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *?7?*2
_class(
&$loc:@dnn/hiddenlayer_4/kernel/part_0
?
Hdnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:

*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_4/kernel/part_0*
seed2 
?
>dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_4/kernel/part_0
?
>dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes

:

*
T0*2
_class(
&$loc:@dnn/hiddenlayer_4/kernel/part_0
?
:dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:

*
T0*2
_class(
&$loc:@dnn/hiddenlayer_4/kernel/part_0
?
dnn/hiddenlayer_4/kernel/part_0VarHandleOp*0
shared_name!dnn/hiddenlayer_4/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_4/kernel/part_0*
	container *
shape
:

*
dtype0*
_output_shapes
: 
?
@dnn/hiddenlayer_4/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_4/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_4/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_4/kernel/part_0:dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_4/kernel/part_0
?
3dnn/hiddenlayer_4/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_4/kernel/part_0*
dtype0*
_output_shapes

:


?
/dnn/hiddenlayer_4/bias/part_0/Initializer/zerosConst*
valueB
*    *0
_class&
$"loc:@dnn/hiddenlayer_4/bias/part_0*
dtype0*
_output_shapes
:

?
dnn/hiddenlayer_4/bias/part_0VarHandleOp*
	container *
shape:
*
dtype0*
_output_shapes
: *.
shared_namednn/hiddenlayer_4/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_4/bias/part_0
?
>dnn/hiddenlayer_4/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_4/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_4/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_4/bias/part_0/dnn/hiddenlayer_4/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_4/bias/part_0*
dtype0
?
1dnn/hiddenlayer_4/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/bias/part_0*
dtype0*
_output_shapes
:
*0
_class&
$"loc:@dnn/hiddenlayer_4/bias/part_0
?
'dnn/hiddenlayer_4/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/kernel/part_0*
dtype0*
_output_shapes

:


v
dnn/hiddenlayer_4/kernelIdentity'dnn/hiddenlayer_4/kernel/ReadVariableOp*
T0*
_output_shapes

:


?
dnn/hiddenlayer_4/MatMulMatMuldnn/hiddenlayer_3/Reludnn/hiddenlayer_4/kernel*
_output_shapes

: 
*
transpose_a( *
transpose_b( *
T0

%dnn/hiddenlayer_4/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/bias/part_0*
dtype0*
_output_shapes
:

n
dnn/hiddenlayer_4/biasIdentity%dnn/hiddenlayer_4/bias/ReadVariableOp*
_output_shapes
:
*
T0
?
dnn/hiddenlayer_4/BiasAddBiasAdddnn/hiddenlayer_4/MatMuldnn/hiddenlayer_4/bias*
T0*
data_formatNHWC*
_output_shapes

: 

`
dnn/hiddenlayer_4/ReluReludnn/hiddenlayer_4/BiasAdd*
_output_shapes

: 
*
T0
Z
dnn/zero_fraction_4/SizeConst*
dtype0	*
_output_shapes
: *
value	B	 R 
e
dnn/zero_fraction_4/LessEqual/yConst*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
dnn/zero_fraction_4/LessEqual	LessEqualdnn/zero_fraction_4/Sizednn/zero_fraction_4/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_4/cond/SwitchSwitchdnn/zero_fraction_4/LessEqualdnn/zero_fraction_4/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_4/cond/switch_tIdentity!dnn/zero_fraction_4/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_4/cond/switch_fIdentitydnn/zero_fraction_4/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_4/cond/pred_idIdentitydnn/zero_fraction_4/LessEqual*
_output_shapes
: *
T0

?
,dnn/zero_fraction_4/cond/count_nonzero/zerosConst"^dnn/zero_fraction_4/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
/dnn/zero_fraction_4/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_4/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_4/cond/count_nonzero/zeros*
T0*
_output_shapes

: 

?
6dnn/zero_fraction_4/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_4/Relu dnn/zero_fraction_4/cond/pred_id*$
_output_shapes
: 
: 
*
T0*)
_class
loc:@dnn/hiddenlayer_4/Relu
?
+dnn/zero_fraction_4/cond/count_nonzero/CastCast/dnn/zero_fraction_4/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

: 
*

DstT0
?
,dnn/zero_fraction_4/cond/count_nonzero/ConstConst"^dnn/zero_fraction_4/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
?
4dnn/zero_fraction_4/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_4/cond/count_nonzero/Cast,dnn/zero_fraction_4/cond/count_nonzero/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
?
dnn/zero_fraction_4/cond/CastCast4dnn/zero_fraction_4/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
?
.dnn/zero_fraction_4/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_4/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
1dnn/zero_fraction_4/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_4/cond/count_nonzero_1/zeros*
T0*
_output_shapes

: 

?
8dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_4/Relu dnn/zero_fraction_4/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_4/Relu*$
_output_shapes
: 
: 

?
-dnn/zero_fraction_4/cond/count_nonzero_1/CastCast1dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual*
Truncate( *
_output_shapes

: 
*

DstT0	*

SrcT0

?
.dnn/zero_fraction_4/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_4/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
6dnn/zero_fraction_4/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_4/cond/count_nonzero_1/Cast.dnn/zero_fraction_4/cond/count_nonzero_1/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_4/cond/MergeMerge6dnn/zero_fraction_4/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_4/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
*dnn/zero_fraction_4/counts_to_fraction/subSubdnn/zero_fraction_4/Sizednn/zero_fraction_4/cond/Merge*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_4/counts_to_fraction/CastCast*dnn/zero_fraction_4/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
-dnn/zero_fraction_4/counts_to_fraction/Cast_1Castdnn/zero_fraction_4/Size*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
.dnn/zero_fraction_4/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_4/counts_to_fraction/Cast-dnn/zero_fraction_4/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_4/fractionIdentity.dnn/zero_fraction_4/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2dnn/dnn/hiddenlayer_4/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_4/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
-dnn/dnn/hiddenlayer_4/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_4/fraction_of_zero_values/tagsdnn/zero_fraction_4/fraction*
T0*
_output_shapes
: 
?
$dnn/dnn/hiddenlayer_4/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_4/activation*
dtype0*
_output_shapes
: 
?
 dnn/dnn/hiddenlayer_4/activationHistogramSummary$dnn/dnn/hiddenlayer_4/activation/tagdnn/hiddenlayer_4/Relu*
T0*
_output_shapes
: 
?
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
:
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *?=?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *?=?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
?
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:
*

seed 
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:

?
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:

?
dnn/logits/kernel/part_0VarHandleOp*)
shared_namednn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape
:
*
dtype0*
_output_shapes
: 
?
9dnn/logits/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel/part_0*
_output_shapes
: 
?
dnn/logits/kernel/part_0/AssignAssignVariableOpdnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
?
,dnn/logits/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:

?
(dnn/logits/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *)
_class
loc:@dnn/logits/bias/part_0
?
dnn/logits/bias/part_0VarHandleOp*
shape:*
dtype0*
_output_shapes
: *'
shared_namednn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
	container 
}
7dnn/logits/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias/part_0*
_output_shapes
: 
?
dnn/logits/bias/part_0/AssignAssignVariableOpdnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
dtype0*)
_class
loc:@dnn/logits/bias/part_0
?
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

:

h
dnn/logits/kernelIdentity dnn/logits/kernel/ReadVariableOp*
T0*
_output_shapes

:

?
dnn/logits/MatMulMatMuldnn/hiddenlayer_4/Reludnn/logits/kernel*
_output_shapes

: *
transpose_a( *
transpose_b( *
T0
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

dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*
data_formatNHWC*
_output_shapes

: 
Z
dnn/zero_fraction_5/SizeConst*
value	B	 R *
dtype0	*
_output_shapes
: 
e
dnn/zero_fraction_5/LessEqual/yConst*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
dnn/zero_fraction_5/LessEqual	LessEqualdnn/zero_fraction_5/Sizednn/zero_fraction_5/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_5/cond/SwitchSwitchdnn/zero_fraction_5/LessEqualdnn/zero_fraction_5/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_5/cond/switch_tIdentity!dnn/zero_fraction_5/cond/Switch:1*
_output_shapes
: *
T0

o
!dnn/zero_fraction_5/cond/switch_fIdentitydnn/zero_fraction_5/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_5/cond/pred_idIdentitydnn/zero_fraction_5/LessEqual*
T0
*
_output_shapes
: 
?
,dnn/zero_fraction_5/cond/count_nonzero/zerosConst"^dnn/zero_fraction_5/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
/dnn/zero_fraction_5/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_5/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_5/cond/count_nonzero/zeros*
T0*
_output_shapes

: 
?
6dnn/zero_fraction_5/cond/count_nonzero/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_5/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*$
_output_shapes
: : 
?
+dnn/zero_fraction_5/cond/count_nonzero/CastCast/dnn/zero_fraction_5/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

: *

DstT0
?
,dnn/zero_fraction_5/cond/count_nonzero/ConstConst"^dnn/zero_fraction_5/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
4dnn/zero_fraction_5/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_5/cond/count_nonzero/Cast,dnn/zero_fraction_5/cond/count_nonzero/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
?
dnn/zero_fraction_5/cond/CastCast4dnn/zero_fraction_5/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
?
.dnn/zero_fraction_5/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_5/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
1dnn/zero_fraction_5/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_5/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_5/cond/count_nonzero_1/zeros*
_output_shapes

: *
T0
?
8dnn/zero_fraction_5/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_5/cond/pred_id*$
_output_shapes
: : *
T0*%
_class
loc:@dnn/logits/BiasAdd
?
-dnn/zero_fraction_5/cond/count_nonzero_1/CastCast1dnn/zero_fraction_5/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

: *

DstT0	
?
.dnn/zero_fraction_5/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_5/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
6dnn/zero_fraction_5/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_5/cond/count_nonzero_1/Cast.dnn/zero_fraction_5/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: *
	keep_dims( *

Tidx0
?
dnn/zero_fraction_5/cond/MergeMerge6dnn/zero_fraction_5/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_5/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
*dnn/zero_fraction_5/counts_to_fraction/subSubdnn/zero_fraction_5/Sizednn/zero_fraction_5/cond/Merge*
_output_shapes
: *
T0	
?
+dnn/zero_fraction_5/counts_to_fraction/CastCast*dnn/zero_fraction_5/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
-dnn/zero_fraction_5/counts_to_fraction/Cast_1Castdnn/zero_fraction_5/Size*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
.dnn/zero_fraction_5/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_5/counts_to_fraction/Cast-dnn/zero_fraction_5/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_5/fractionIdentity.dnn/zero_fraction_5/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_5/fraction*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
?
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
f
dnn/head/logits/ShapeConst*
valueB"       *
dtype0*
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
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
r
save/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:

X
save/IdentityIdentitysave/Read/ReadVariableOp*
T0*
_output_shapes
:

^
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes
:

z
save/Read_1/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:

`
save/Identity_2Identitysave/Read_1/ReadVariableOp*
_output_shapes

:
*
T0
d
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
T0*
_output_shapes

:

t
save/Read_2/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:

\
save/Identity_4Identitysave/Read_2/ReadVariableOp*
T0*
_output_shapes
:

`
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
T0*
_output_shapes
:

z
save/Read_3/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:


`
save/Identity_6Identitysave/Read_3/ReadVariableOp*
T0*
_output_shapes

:


d
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
T0*
_output_shapes

:


t
save/Read_4/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
:

\
save/Identity_8Identitysave/Read_4/ReadVariableOp*
T0*
_output_shapes
:

`
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
_output_shapes
:
*
T0
z
save/Read_5/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes

:


a
save/Identity_10Identitysave/Read_5/ReadVariableOp*
_output_shapes

:

*
T0
f
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes

:


t
save/Read_6/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/bias/part_0*
dtype0*
_output_shapes
:

]
save/Identity_12Identitysave/Read_6/ReadVariableOp*
T0*
_output_shapes
:

b
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
T0*
_output_shapes
:

z
save/Read_7/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes

:


a
save/Identity_14Identitysave/Read_7/ReadVariableOp*
T0*
_output_shapes

:


f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
_output_shapes

:

*
T0
t
save/Read_8/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/bias/part_0*
dtype0*
_output_shapes
:

]
save/Identity_16Identitysave/Read_8/ReadVariableOp*
_output_shapes
:
*
T0
b
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
_output_shapes
:
*
T0
z
save/Read_9/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/kernel/part_0*
dtype0*
_output_shapes

:


a
save/Identity_18Identitysave/Read_9/ReadVariableOp*
T0*
_output_shapes

:


f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
T0*
_output_shapes

:


n
save/Read_10/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
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
t
save/Read_11/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:

b
save/Identity_22Identitysave/Read_11/ReadVariableOp*
T0*
_output_shapes

:

f
save/Identity_23Identitysave/Identity_22"/device:CPU:0*
T0*
_output_shapes

:

?
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_8482c5c746334b10a06652117ba76f8b/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
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
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
{
save/SaveV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
dtype0*
_output_shapes
:
t
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step"/device:CPU:0*
dtypes
2	
?
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
?
save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/Read_12/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:

m
save/Identity_24Identitysave/Read_12/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:

b
save/Identity_25Identitysave/Identity_24"/device:CPU:0*
T0*
_output_shapes
:

?
save/Read_13/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:

q
save/Identity_26Identitysave/Read_13/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:

f
save/Identity_27Identitysave/Identity_26"/device:CPU:0*
T0*
_output_shapes

:

?
save/Read_14/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:

m
save/Identity_28Identitysave/Read_14/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:

b
save/Identity_29Identitysave/Identity_28"/device:CPU:0*
_output_shapes
:
*
T0
?
save/Read_15/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:


q
save/Identity_30Identitysave/Read_15/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:


f
save/Identity_31Identitysave/Identity_30"/device:CPU:0*
_output_shapes

:

*
T0
?
save/Read_16/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:

m
save/Identity_32Identitysave/Read_16/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:

b
save/Identity_33Identitysave/Identity_32"/device:CPU:0*
T0*
_output_shapes
:

?
save/Read_17/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:


q
save/Identity_34Identitysave/Read_17/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:


f
save/Identity_35Identitysave/Identity_34"/device:CPU:0*
T0*
_output_shapes

:


?
save/Read_18/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:

m
save/Identity_36Identitysave/Read_18/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:

b
save/Identity_37Identitysave/Identity_36"/device:CPU:0*
T0*
_output_shapes
:

?
save/Read_19/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:


q
save/Identity_38Identitysave/Read_19/ReadVariableOp"/device:CPU:0*
_output_shapes

:

*
T0
f
save/Identity_39Identitysave/Identity_38"/device:CPU:0*
T0*
_output_shapes

:


?
save/Read_20/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:

m
save/Identity_40Identitysave/Read_20/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:

b
save/Identity_41Identitysave/Identity_40"/device:CPU:0*
_output_shapes
:
*
T0
?
save/Read_21/ReadVariableOpReadVariableOpdnn/hiddenlayer_4/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:


q
save/Identity_42Identitysave/Read_21/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:


f
save/Identity_43Identitysave/Identity_42"/device:CPU:0*
_output_shapes

:

*
T0
}
save/Read_22/ReadVariableOpReadVariableOpdnn/logits/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_44Identitysave/Read_22/ReadVariableOp"/device:CPU:0*
_output_shapes
:*
T0
b
save/Identity_45Identitysave/Identity_44"/device:CPU:0*
T0*
_output_shapes
:
?
save/Read_23/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:

q
save/Identity_46Identitysave/Read_23/ReadVariableOp"/device:CPU:0*
_output_shapes

:
*
T0
f
save/Identity_47Identitysave/Identity_46"/device:CPU:0*
T0*
_output_shapes

:

?
save/SaveV2_1/tensor_namesConst"/device:CPU:0*?
value?B?Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/hiddenlayer_3/biasBdnn/hiddenlayer_3/kernelBdnn/hiddenlayer_4/biasBdnn/hiddenlayer_4/kernelBdnn/logits/biasBdnn/logits/kernel*
dtype0*
_output_shapes
:
?
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*?
value?B?B10 0,10B6 10 0,6:0,10B10 0,10B10 10 0,10:0,10B10 0,10B10 10 0,10:0,10B10 0,10B10 10 0,10:0,10B10 0,10B10 10 0,10:0,10B1 0,1B10 1 0,10:0,1*
dtype0*
_output_shapes
:
?
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_25save/Identity_27save/Identity_29save/Identity_31save/Identity_33save/Identity_35save/Identity_37save/Identity_39save/Identity_41save/Identity_43save/Identity_45save/Identity_47"/device:CPU:0*
dtypes
2
?
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
?
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
?
save/Identity_48Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
~
save/RestoreV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
dtype0*
_output_shapes
:
w
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2	
?
save/AssignAssignglobal_stepsave/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
(
save/restore_shardNoOp^save/Assign
?
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*?
value?B?Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/hiddenlayer_3/biasBdnn/hiddenlayer_3/kernelBdnn/hiddenlayer_4/biasBdnn/hiddenlayer_4/kernelBdnn/logits/biasBdnn/logits/kernel
?
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*?
value?B?B10 0,10B6 10 0,6:0,10B10 0,10B10 10 0,10:0,10B10 0,10B10 10 0,10:0,10B10 0,10B10 10 0,10:0,10B10 0,10B10 10 0,10:0,10B1 0,1B10 1 0,10:0,1*
dtype0*
_output_shapes
:
?
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*t
_output_shapesb
`:
:
:
:

:
:

:
:

:
:

::
*
dtypes
2
b
save/Identity_49Identitysave/RestoreV2_1"/device:CPU:0*
T0*
_output_shapes
:

v
save/AssignVariableOpAssignVariableOpdnn/hiddenlayer_0/bias/part_0save/Identity_49"/device:CPU:0*
dtype0
h
save/Identity_50Identitysave/RestoreV2_1:1"/device:CPU:0*
T0*
_output_shapes

:

z
save/AssignVariableOp_1AssignVariableOpdnn/hiddenlayer_0/kernel/part_0save/Identity_50"/device:CPU:0*
dtype0
d
save/Identity_51Identitysave/RestoreV2_1:2"/device:CPU:0*
T0*
_output_shapes
:

x
save/AssignVariableOp_2AssignVariableOpdnn/hiddenlayer_1/bias/part_0save/Identity_51"/device:CPU:0*
dtype0
h
save/Identity_52Identitysave/RestoreV2_1:3"/device:CPU:0*
T0*
_output_shapes

:


z
save/AssignVariableOp_3AssignVariableOpdnn/hiddenlayer_1/kernel/part_0save/Identity_52"/device:CPU:0*
dtype0
d
save/Identity_53Identitysave/RestoreV2_1:4"/device:CPU:0*
T0*
_output_shapes
:

x
save/AssignVariableOp_4AssignVariableOpdnn/hiddenlayer_2/bias/part_0save/Identity_53"/device:CPU:0*
dtype0
h
save/Identity_54Identitysave/RestoreV2_1:5"/device:CPU:0*
T0*
_output_shapes

:


z
save/AssignVariableOp_5AssignVariableOpdnn/hiddenlayer_2/kernel/part_0save/Identity_54"/device:CPU:0*
dtype0
d
save/Identity_55Identitysave/RestoreV2_1:6"/device:CPU:0*
T0*
_output_shapes
:

x
save/AssignVariableOp_6AssignVariableOpdnn/hiddenlayer_3/bias/part_0save/Identity_55"/device:CPU:0*
dtype0
h
save/Identity_56Identitysave/RestoreV2_1:7"/device:CPU:0*
T0*
_output_shapes

:


z
save/AssignVariableOp_7AssignVariableOpdnn/hiddenlayer_3/kernel/part_0save/Identity_56"/device:CPU:0*
dtype0
d
save/Identity_57Identitysave/RestoreV2_1:8"/device:CPU:0*
_output_shapes
:
*
T0
x
save/AssignVariableOp_8AssignVariableOpdnn/hiddenlayer_4/bias/part_0save/Identity_57"/device:CPU:0*
dtype0
h
save/Identity_58Identitysave/RestoreV2_1:9"/device:CPU:0*
_output_shapes

:

*
T0
z
save/AssignVariableOp_9AssignVariableOpdnn/hiddenlayer_4/kernel/part_0save/Identity_58"/device:CPU:0*
dtype0
e
save/Identity_59Identitysave/RestoreV2_1:10"/device:CPU:0*
T0*
_output_shapes
:
r
save/AssignVariableOp_10AssignVariableOpdnn/logits/bias/part_0save/Identity_59"/device:CPU:0*
dtype0
i
save/Identity_60Identitysave/RestoreV2_1:11"/device:CPU:0*
_output_shapes

:
*
T0
t
save/AssignVariableOp_11AssignVariableOpdnn/logits/kernel/part_0save/Identity_60"/device:CPU:0*
dtype0
?
save/restore_shard_1NoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"?
save/Const:0save/Identity_48:0save/restore_all (5 @F8"?
trainable_variables??
?
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernel
  "
(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias
 "
(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel

  "

(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias
 "
(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign5dnn/hiddenlayer_2/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_2/kernel

  "

(2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign3dnn/hiddenlayer_2/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_2/bias
 "
(21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_3/kernel/part_0:0&dnn/hiddenlayer_3/kernel/part_0/Assign5dnn/hiddenlayer_3/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_3/kernel

  "

(2<dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_3/bias/part_0:0$dnn/hiddenlayer_3/bias/part_0/Assign3dnn/hiddenlayer_3/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_3/bias
 "
(21dnn/hiddenlayer_3/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_4/kernel/part_0:0&dnn/hiddenlayer_4/kernel/part_0/Assign5dnn/hiddenlayer_4/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_4/kernel

  "

(2<dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_4/bias/part_0:0$dnn/hiddenlayer_4/bias/part_0/Assign3dnn/hiddenlayer_4/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_4/bias
 "
(21dnn/hiddenlayer_4/bias/part_0/Initializer/zeros:08
?
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel
  "
(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
?
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08"?
	summaries?
?
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
/dnn/dnn/hiddenlayer_2/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_2/activation:0
/dnn/dnn/hiddenlayer_3/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_3/activation:0
/dnn/dnn/hiddenlayer_4/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_4/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"?
	variables??
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
?
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernel
  "
(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias
 "
(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel

  "

(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias
 "
(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign5dnn/hiddenlayer_2/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_2/kernel

  "

(2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign3dnn/hiddenlayer_2/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_2/bias
 "
(21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_3/kernel/part_0:0&dnn/hiddenlayer_3/kernel/part_0/Assign5dnn/hiddenlayer_3/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_3/kernel

  "

(2<dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_3/bias/part_0:0$dnn/hiddenlayer_3/bias/part_0/Assign3dnn/hiddenlayer_3/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_3/bias
 "
(21dnn/hiddenlayer_3/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_4/kernel/part_0:0&dnn/hiddenlayer_4/kernel/part_0/Assign5dnn/hiddenlayer_4/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_4/kernel

  "

(2<dnn/hiddenlayer_4/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_4/bias/part_0:0$dnn/hiddenlayer_4/bias/part_0/Assign3dnn/hiddenlayer_4/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_4/bias
 "
(21dnn/hiddenlayer_4/bias/part_0/Initializer/zeros:08
?
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel
  "
(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
?
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"?A
cond_context?A?A
?
 dnn/zero_fraction/cond/cond_text dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_t:0 *?
dnn/hiddenlayer_0/Relu:0
dnn/zero_fraction/cond/Cast:0
+dnn/zero_fraction/cond/count_nonzero/Cast:0
,dnn/zero_fraction/cond/count_nonzero/Const:0
6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
/dnn/zero_fraction/cond/count_nonzero/NotEqual:0
4dnn/zero_fraction/cond/count_nonzero/nonzero_count:0
,dnn/zero_fraction/cond/count_nonzero/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_t:0R
dnn/hiddenlayer_0/Relu:06dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0
?
"dnn/zero_fraction/cond/cond_text_1 dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_f:0*?
dnn/hiddenlayer_0/Relu:0
-dnn/zero_fraction/cond/count_nonzero_1/Cast:0
.dnn/zero_fraction/cond/count_nonzero_1/Const:0
8dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
1dnn/zero_fraction/cond/count_nonzero_1/NotEqual:0
6dnn/zero_fraction/cond/count_nonzero_1/nonzero_count:0
.dnn/zero_fraction/cond/count_nonzero_1/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_f:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0T
dnn/hiddenlayer_0/Relu:08dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
?
"dnn/zero_fraction_1/cond/cond_text"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_t:0 *?
dnn/hiddenlayer_1/Relu:0
dnn/zero_fraction_1/cond/Cast:0
-dnn/zero_fraction_1/cond/count_nonzero/Cast:0
.dnn/zero_fraction_1/cond/count_nonzero/Const:0
8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_1/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_1/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_1/cond/count_nonzero/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_t:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0T
dnn/hiddenlayer_1/Relu:08dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
?
$dnn/zero_fraction_1/cond/cond_text_1"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_f:0*?
dnn/hiddenlayer_1/Relu:0
/dnn/zero_fraction_1/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_1/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_1/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_f:0V
dnn/hiddenlayer_1/Relu:0:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0
?
"dnn/zero_fraction_2/cond/cond_text"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_t:0 *?
dnn/hiddenlayer_2/Relu:0
dnn/zero_fraction_2/cond/Cast:0
-dnn/zero_fraction_2/cond/count_nonzero/Cast:0
.dnn/zero_fraction_2/cond/count_nonzero/Const:0
8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_2/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_2/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_2/cond/count_nonzero/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_t:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0T
dnn/hiddenlayer_2/Relu:08dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
?
$dnn/zero_fraction_2/cond/cond_text_1"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_f:0*?
dnn/hiddenlayer_2/Relu:0
/dnn/zero_fraction_2/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_2/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_2/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_f:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0V
dnn/hiddenlayer_2/Relu:0:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0
?
"dnn/zero_fraction_3/cond/cond_text"dnn/zero_fraction_3/cond/pred_id:0#dnn/zero_fraction_3/cond/switch_t:0 *?
dnn/hiddenlayer_3/Relu:0
dnn/zero_fraction_3/cond/Cast:0
-dnn/zero_fraction_3/cond/count_nonzero/Cast:0
.dnn/zero_fraction_3/cond/count_nonzero/Const:0
8dnn/zero_fraction_3/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_3/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_3/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_3/cond/count_nonzero/zeros:0
"dnn/zero_fraction_3/cond/pred_id:0
#dnn/zero_fraction_3/cond/switch_t:0T
dnn/hiddenlayer_3/Relu:08dnn/zero_fraction_3/cond/count_nonzero/NotEqual/Switch:1H
"dnn/zero_fraction_3/cond/pred_id:0"dnn/zero_fraction_3/cond/pred_id:0
?
$dnn/zero_fraction_3/cond/cond_text_1"dnn/zero_fraction_3/cond/pred_id:0#dnn/zero_fraction_3/cond/switch_f:0*?
dnn/hiddenlayer_3/Relu:0
/dnn/zero_fraction_3/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_3/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_3/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_3/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_3/cond/pred_id:0
#dnn/zero_fraction_3/cond/switch_f:0H
"dnn/zero_fraction_3/cond/pred_id:0"dnn/zero_fraction_3/cond/pred_id:0V
dnn/hiddenlayer_3/Relu:0:dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/Switch:0
?
"dnn/zero_fraction_4/cond/cond_text"dnn/zero_fraction_4/cond/pred_id:0#dnn/zero_fraction_4/cond/switch_t:0 *?
dnn/hiddenlayer_4/Relu:0
dnn/zero_fraction_4/cond/Cast:0
-dnn/zero_fraction_4/cond/count_nonzero/Cast:0
.dnn/zero_fraction_4/cond/count_nonzero/Const:0
8dnn/zero_fraction_4/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_4/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_4/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_4/cond/count_nonzero/zeros:0
"dnn/zero_fraction_4/cond/pred_id:0
#dnn/zero_fraction_4/cond/switch_t:0H
"dnn/zero_fraction_4/cond/pred_id:0"dnn/zero_fraction_4/cond/pred_id:0T
dnn/hiddenlayer_4/Relu:08dnn/zero_fraction_4/cond/count_nonzero/NotEqual/Switch:1
?
$dnn/zero_fraction_4/cond/cond_text_1"dnn/zero_fraction_4/cond/pred_id:0#dnn/zero_fraction_4/cond/switch_f:0*?
dnn/hiddenlayer_4/Relu:0
/dnn/zero_fraction_4/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_4/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_4/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_4/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_4/cond/pred_id:0
#dnn/zero_fraction_4/cond/switch_f:0H
"dnn/zero_fraction_4/cond/pred_id:0"dnn/zero_fraction_4/cond/pred_id:0V
dnn/hiddenlayer_4/Relu:0:dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual/Switch:0
?
"dnn/zero_fraction_5/cond/cond_text"dnn/zero_fraction_5/cond/pred_id:0#dnn/zero_fraction_5/cond/switch_t:0 *?
dnn/logits/BiasAdd:0
dnn/zero_fraction_5/cond/Cast:0
-dnn/zero_fraction_5/cond/count_nonzero/Cast:0
.dnn/zero_fraction_5/cond/count_nonzero/Const:0
8dnn/zero_fraction_5/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_5/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_5/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_5/cond/count_nonzero/zeros:0
"dnn/zero_fraction_5/cond/pred_id:0
#dnn/zero_fraction_5/cond/switch_t:0H
"dnn/zero_fraction_5/cond/pred_id:0"dnn/zero_fraction_5/cond/pred_id:0P
dnn/logits/BiasAdd:08dnn/zero_fraction_5/cond/count_nonzero/NotEqual/Switch:1
?
$dnn/zero_fraction_5/cond/cond_text_1"dnn/zero_fraction_5/cond/pred_id:0#dnn/zero_fraction_5/cond/switch_f:0*?
dnn/logits/BiasAdd:0
/dnn/zero_fraction_5/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_5/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_5/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_5/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_5/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_5/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_5/cond/pred_id:0
#dnn/zero_fraction_5/cond/switch_f:0H
"dnn/zero_fraction_5/cond/pred_id:0"dnn/zero_fraction_5/cond/pred_id:0R
dnn/logits/BiasAdd:0:dnn/zero_fraction_5/cond/count_nonzero_1/NotEqual/Switch:0"%
saved_model_main_op


group_deps*?
serving_defaults
(
inputs
input_example_tensor:0+
outputs 
dnn/logits/BiasAdd:0 tensorflow/serving/regress*?
predicty
*
examples
input_example_tensor:0/
predictions 
dnn/logits/BiasAdd:0 tensorflow/serving/predict*?

regressions
(
inputs
input_example_tensor:0+
outputs 
dnn/logits/BiasAdd:0 tensorflow/serving/regress