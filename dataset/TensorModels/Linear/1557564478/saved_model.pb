??
??
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
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
?
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
b'unknown'??
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
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container *
shape: 
?
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
Y
input_example_tensorPlaceholder*
shape:*
dtype0*
_output_shapes
:
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
ParseExample/Const_2Const*
valueB	 *
dtype0	*
_output_shapes
: 
W
ParseExample/Const_3Const*
dtype0	*
_output_shapes
: *
valueB	 
W
ParseExample/Const_4Const*
valueB	 *
dtype0	*
_output_shapes
: 
W
ParseExample/Const_5Const*
dtype0*
_output_shapes
: *
valueB 
b
ParseExample/ParseExample/namesConst*
dtype0*
_output_shapes
: *
valueB 
p
&ParseExample/ParseExample/dense_keys_0Const*
dtype0*
_output_shapes
: *
valueB B	drug1_prr
p
&ParseExample/ParseExample/dense_keys_1Const*
valueB B	drug2_prr*
dtype0*
_output_shapes
: 
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
&ParseExample/ParseExample/dense_keys_4Const*
dtype0*
_output_shapes
: *
valueB Bevent
j
&ParseExample/ParseExample/dense_keys_5Const*
valueB	 Bprr*
dtype0*
_output_shapes
: 
?
ParseExample/ParseExampleParseExampleinput_example_tensorParseExample/ParseExample/names&ParseExample/ParseExample/dense_keys_0&ParseExample/ParseExample/dense_keys_1&ParseExample/ParseExample/dense_keys_2&ParseExample/ParseExample/dense_keys_3&ParseExample/ParseExample/dense_keys_4&ParseExample/ParseExample/dense_keys_5ParseExample/ConstParseExample/Const_1ParseExample/Const_2ParseExample/Const_3ParseExample/Const_4ParseExample/Const_5*
Ndense*z
_output_shapesh
f:????????? :????????? :????????? :????????? :????????? :????????? *
Nsparse *
sparse_types
 **
dense_shapes
: : : : : : *
Tdense

2			
?
>linear/linear_model/drug1_prr/weights/part_0/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *?
_class5
31loc:@linear/linear_model/drug1_prr/weights/part_0
?
,linear/linear_model/drug1_prr/weights/part_0VarHandleOp*=
shared_name.,linear/linear_model/drug1_prr/weights/part_0*?
_class5
31loc:@linear/linear_model/drug1_prr/weights/part_0*
	container *
shape
:*
dtype0*
_output_shapes
: 
?
Mlinear/linear_model/drug1_prr/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/drug1_prr/weights/part_0*
_output_shapes
: 
?
3linear/linear_model/drug1_prr/weights/part_0/AssignAssignVariableOp,linear/linear_model/drug1_prr/weights/part_0>linear/linear_model/drug1_prr/weights/part_0/Initializer/zeros*?
_class5
31loc:@linear/linear_model/drug1_prr/weights/part_0*
dtype0
?
@linear/linear_model/drug1_prr/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/drug1_prr/weights/part_0*?
_class5
31loc:@linear/linear_model/drug1_prr/weights/part_0*
dtype0*
_output_shapes

:
?
>linear/linear_model/drug2_prr/weights/part_0/Initializer/zerosConst*
valueB*    *?
_class5
31loc:@linear/linear_model/drug2_prr/weights/part_0*
dtype0*
_output_shapes

:
?
,linear/linear_model/drug2_prr/weights/part_0VarHandleOp*
dtype0*
_output_shapes
: *=
shared_name.,linear/linear_model/drug2_prr/weights/part_0*?
_class5
31loc:@linear/linear_model/drug2_prr/weights/part_0*
	container *
shape
:
?
Mlinear/linear_model/drug2_prr/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/drug2_prr/weights/part_0*
_output_shapes
: 
?
3linear/linear_model/drug2_prr/weights/part_0/AssignAssignVariableOp,linear/linear_model/drug2_prr/weights/part_0>linear/linear_model/drug2_prr/weights/part_0/Initializer/zeros*?
_class5
31loc:@linear/linear_model/drug2_prr/weights/part_0*
dtype0
?
@linear/linear_model/drug2_prr/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/drug2_prr/weights/part_0*?
_class5
31loc:@linear/linear_model/drug2_prr/weights/part_0*
dtype0*
_output_shapes

:
?
;linear/linear_model/drug_1/weights/part_0/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@linear/linear_model/drug_1/weights/part_0*
dtype0*
_output_shapes

:
?
)linear/linear_model/drug_1/weights/part_0VarHandleOp*<
_class2
0.loc:@linear/linear_model/drug_1/weights/part_0*
	container *
shape
:*
dtype0*
_output_shapes
: *:
shared_name+)linear/linear_model/drug_1/weights/part_0
?
Jlinear/linear_model/drug_1/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp)linear/linear_model/drug_1/weights/part_0*
_output_shapes
: 
?
0linear/linear_model/drug_1/weights/part_0/AssignAssignVariableOp)linear/linear_model/drug_1/weights/part_0;linear/linear_model/drug_1/weights/part_0/Initializer/zeros*
dtype0*<
_class2
0.loc:@linear/linear_model/drug_1/weights/part_0
?
=linear/linear_model/drug_1/weights/part_0/Read/ReadVariableOpReadVariableOp)linear/linear_model/drug_1/weights/part_0*<
_class2
0.loc:@linear/linear_model/drug_1/weights/part_0*
dtype0*
_output_shapes

:
?
;linear/linear_model/drug_2/weights/part_0/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *<
_class2
0.loc:@linear/linear_model/drug_2/weights/part_0
?
)linear/linear_model/drug_2/weights/part_0VarHandleOp*
dtype0*
_output_shapes
: *:
shared_name+)linear/linear_model/drug_2/weights/part_0*<
_class2
0.loc:@linear/linear_model/drug_2/weights/part_0*
	container *
shape
:
?
Jlinear/linear_model/drug_2/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp)linear/linear_model/drug_2/weights/part_0*
_output_shapes
: 
?
0linear/linear_model/drug_2/weights/part_0/AssignAssignVariableOp)linear/linear_model/drug_2/weights/part_0;linear/linear_model/drug_2/weights/part_0/Initializer/zeros*<
_class2
0.loc:@linear/linear_model/drug_2/weights/part_0*
dtype0
?
=linear/linear_model/drug_2/weights/part_0/Read/ReadVariableOpReadVariableOp)linear/linear_model/drug_2/weights/part_0*<
_class2
0.loc:@linear/linear_model/drug_2/weights/part_0*
dtype0*
_output_shapes

:
?
:linear/linear_model/event/weights/part_0/Initializer/zerosConst*
valueB*    *;
_class1
/-loc:@linear/linear_model/event/weights/part_0*
dtype0*
_output_shapes

:
?
(linear/linear_model/event/weights/part_0VarHandleOp*
dtype0*
_output_shapes
: *9
shared_name*(linear/linear_model/event/weights/part_0*;
_class1
/-loc:@linear/linear_model/event/weights/part_0*
	container *
shape
:
?
Ilinear/linear_model/event/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp(linear/linear_model/event/weights/part_0*
_output_shapes
: 
?
/linear/linear_model/event/weights/part_0/AssignAssignVariableOp(linear/linear_model/event/weights/part_0:linear/linear_model/event/weights/part_0/Initializer/zeros*;
_class1
/-loc:@linear/linear_model/event/weights/part_0*
dtype0
?
<linear/linear_model/event/weights/part_0/Read/ReadVariableOpReadVariableOp(linear/linear_model/event/weights/part_0*;
_class1
/-loc:@linear/linear_model/event/weights/part_0*
dtype0*
_output_shapes

:
?
8linear/linear_model/prr/weights/part_0/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *9
_class/
-+loc:@linear/linear_model/prr/weights/part_0
?
&linear/linear_model/prr/weights/part_0VarHandleOp*
shape
:*
dtype0*
_output_shapes
: *7
shared_name(&linear/linear_model/prr/weights/part_0*9
_class/
-+loc:@linear/linear_model/prr/weights/part_0*
	container 
?
Glinear/linear_model/prr/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp&linear/linear_model/prr/weights/part_0*
_output_shapes
: 
?
-linear/linear_model/prr/weights/part_0/AssignAssignVariableOp&linear/linear_model/prr/weights/part_08linear/linear_model/prr/weights/part_0/Initializer/zeros*9
_class/
-+loc:@linear/linear_model/prr/weights/part_0*
dtype0
?
:linear/linear_model/prr/weights/part_0/Read/ReadVariableOpReadVariableOp&linear/linear_model/prr/weights/part_0*9
_class/
-+loc:@linear/linear_model/prr/weights/part_0*
dtype0*
_output_shapes

:
?
9linear/linear_model/bias_weights/part_0/Initializer/zerosConst*
valueB*    *:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
?
'linear/linear_model/bias_weights/part_0VarHandleOp*8
shared_name)'linear/linear_model/bias_weights/part_0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
	container *
shape:*
dtype0*
_output_shapes
: 
?
Hlinear/linear_model/bias_weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/bias_weights/part_0*
_output_shapes
: 
?
.linear/linear_model/bias_weights/part_0/AssignAssignVariableOp'linear/linear_model/bias_weights/part_09linear/linear_model/bias_weights/part_0/Initializer/zeros*
dtype0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0
?
;linear/linear_model/bias_weights/part_0/Read/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
?
=linear/linear_model/linear_model/linear_model/drug1_prr/ShapeShapeParseExample/ParseExample*
T0*
out_type0*
_output_shapes
:
?
Klinear/linear_model/linear_model/linear_model/drug1_prr/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Mlinear/linear_model/linear_model/linear_model/drug1_prr/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Mlinear/linear_model/linear_model/linear_model/drug1_prr/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Elinear/linear_model/linear_model/linear_model/drug1_prr/strided_sliceStridedSlice=linear/linear_model/linear_model/linear_model/drug1_prr/ShapeKlinear/linear_model/linear_model/linear_model/drug1_prr/strided_slice/stackMlinear/linear_model/linear_model/linear_model/drug1_prr/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/drug1_prr/strided_slice/stack_2*
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
Glinear/linear_model/linear_model/linear_model/drug1_prr/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
?
Elinear/linear_model/linear_model/linear_model/drug1_prr/Reshape/shapePackElinear/linear_model/linear_model/linear_model/drug1_prr/strided_sliceGlinear/linear_model/linear_model/linear_model/drug1_prr/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
?
?linear/linear_model/linear_model/linear_model/drug1_prr/ReshapeReshapeParseExample/ParseExampleElinear/linear_model/linear_model/linear_model/drug1_prr/Reshape/shape*
T0*
Tshape0*
_output_shapes

: 
?
4linear/linear_model/drug1_prr/weights/ReadVariableOpReadVariableOp,linear/linear_model/drug1_prr/weights/part_0*
dtype0*
_output_shapes

:
?
%linear/linear_model/drug1_prr/weightsIdentity4linear/linear_model/drug1_prr/weights/ReadVariableOp*
T0*
_output_shapes

:
?
Dlinear/linear_model/linear_model/linear_model/drug1_prr/weighted_sumMatMul?linear/linear_model/linear_model/linear_model/drug1_prr/Reshape%linear/linear_model/drug1_prr/weights*
transpose_b( *
T0*
_output_shapes

: *
transpose_a( 
?
=linear/linear_model/linear_model/linear_model/drug2_prr/ShapeShapeParseExample/ParseExample:1*
T0*
out_type0*
_output_shapes
:
?
Klinear/linear_model/linear_model/linear_model/drug2_prr/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Mlinear/linear_model/linear_model/linear_model/drug2_prr/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Mlinear/linear_model/linear_model/linear_model/drug2_prr/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Elinear/linear_model/linear_model/linear_model/drug2_prr/strided_sliceStridedSlice=linear/linear_model/linear_model/linear_model/drug2_prr/ShapeKlinear/linear_model/linear_model/linear_model/drug2_prr/strided_slice/stackMlinear/linear_model/linear_model/linear_model/drug2_prr/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/drug2_prr/strided_slice/stack_2*
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
Glinear/linear_model/linear_model/linear_model/drug2_prr/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Elinear/linear_model/linear_model/linear_model/drug2_prr/Reshape/shapePackElinear/linear_model/linear_model/linear_model/drug2_prr/strided_sliceGlinear/linear_model/linear_model/linear_model/drug2_prr/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
?
?linear/linear_model/linear_model/linear_model/drug2_prr/ReshapeReshapeParseExample/ParseExample:1Elinear/linear_model/linear_model/linear_model/drug2_prr/Reshape/shape*
T0*
Tshape0*
_output_shapes

: 
?
4linear/linear_model/drug2_prr/weights/ReadVariableOpReadVariableOp,linear/linear_model/drug2_prr/weights/part_0*
dtype0*
_output_shapes

:
?
%linear/linear_model/drug2_prr/weightsIdentity4linear/linear_model/drug2_prr/weights/ReadVariableOp*
T0*
_output_shapes

:
?
Dlinear/linear_model/linear_model/linear_model/drug2_prr/weighted_sumMatMul?linear/linear_model/linear_model/linear_model/drug2_prr/Reshape%linear/linear_model/drug2_prr/weights*
T0*
_output_shapes

: *
transpose_a( *
transpose_b( 
?
<linear/linear_model/linear_model/linear_model/drug_1/ToFloatCastParseExample/ParseExample:2*
Truncate( *%
_output_shapes
:????????? *

DstT0*

SrcT0	
?
:linear/linear_model/linear_model/linear_model/drug_1/ShapeShape<linear/linear_model/linear_model/linear_model/drug_1/ToFloat*
_output_shapes
:*
T0*
out_type0
?
Hlinear/linear_model/linear_model/linear_model/drug_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/drug_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/drug_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Blinear/linear_model/linear_model/linear_model/drug_1/strided_sliceStridedSlice:linear/linear_model/linear_model/linear_model/drug_1/ShapeHlinear/linear_model/linear_model/linear_model/drug_1/strided_slice/stackJlinear/linear_model/linear_model/linear_model/drug_1/strided_slice/stack_1Jlinear/linear_model/linear_model/linear_model/drug_1/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
?
Dlinear/linear_model/linear_model/linear_model/drug_1/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Blinear/linear_model/linear_model/linear_model/drug_1/Reshape/shapePackBlinear/linear_model/linear_model/linear_model/drug_1/strided_sliceDlinear/linear_model/linear_model/linear_model/drug_1/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
?
<linear/linear_model/linear_model/linear_model/drug_1/ReshapeReshape<linear/linear_model/linear_model/linear_model/drug_1/ToFloatBlinear/linear_model/linear_model/linear_model/drug_1/Reshape/shape*
T0*
Tshape0*
_output_shapes

: 
?
1linear/linear_model/drug_1/weights/ReadVariableOpReadVariableOp)linear/linear_model/drug_1/weights/part_0*
dtype0*
_output_shapes

:
?
"linear/linear_model/drug_1/weightsIdentity1linear/linear_model/drug_1/weights/ReadVariableOp*
T0*
_output_shapes

:
?
Alinear/linear_model/linear_model/linear_model/drug_1/weighted_sumMatMul<linear/linear_model/linear_model/linear_model/drug_1/Reshape"linear/linear_model/drug_1/weights*
T0*
_output_shapes

: *
transpose_a( *
transpose_b( 
?
<linear/linear_model/linear_model/linear_model/drug_2/ToFloatCastParseExample/ParseExample:3*

SrcT0	*
Truncate( *%
_output_shapes
:????????? *

DstT0
?
:linear/linear_model/linear_model/linear_model/drug_2/ShapeShape<linear/linear_model/linear_model/linear_model/drug_2/ToFloat*
T0*
out_type0*
_output_shapes
:
?
Hlinear/linear_model/linear_model/linear_model/drug_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/drug_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/drug_2/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Blinear/linear_model/linear_model/linear_model/drug_2/strided_sliceStridedSlice:linear/linear_model/linear_model/linear_model/drug_2/ShapeHlinear/linear_model/linear_model/linear_model/drug_2/strided_slice/stackJlinear/linear_model/linear_model/linear_model/drug_2/strided_slice/stack_1Jlinear/linear_model/linear_model/linear_model/drug_2/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
?
Dlinear/linear_model/linear_model/linear_model/drug_2/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Blinear/linear_model/linear_model/linear_model/drug_2/Reshape/shapePackBlinear/linear_model/linear_model/linear_model/drug_2/strided_sliceDlinear/linear_model/linear_model/linear_model/drug_2/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
?
<linear/linear_model/linear_model/linear_model/drug_2/ReshapeReshape<linear/linear_model/linear_model/linear_model/drug_2/ToFloatBlinear/linear_model/linear_model/linear_model/drug_2/Reshape/shape*
T0*
Tshape0*
_output_shapes

: 
?
1linear/linear_model/drug_2/weights/ReadVariableOpReadVariableOp)linear/linear_model/drug_2/weights/part_0*
dtype0*
_output_shapes

:
?
"linear/linear_model/drug_2/weightsIdentity1linear/linear_model/drug_2/weights/ReadVariableOp*
T0*
_output_shapes

:
?
Alinear/linear_model/linear_model/linear_model/drug_2/weighted_sumMatMul<linear/linear_model/linear_model/linear_model/drug_2/Reshape"linear/linear_model/drug_2/weights*
T0*
_output_shapes

: *
transpose_a( *
transpose_b( 
?
;linear/linear_model/linear_model/linear_model/event/ToFloatCastParseExample/ParseExample:4*

SrcT0	*
Truncate( *%
_output_shapes
:????????? *

DstT0
?
9linear/linear_model/linear_model/linear_model/event/ShapeShape;linear/linear_model/linear_model/linear_model/event/ToFloat*
_output_shapes
:*
T0*
out_type0
?
Glinear/linear_model/linear_model/linear_model/event/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/event/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/event/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Alinear/linear_model/linear_model/linear_model/event/strided_sliceStridedSlice9linear/linear_model/linear_model/linear_model/event/ShapeGlinear/linear_model/linear_model/linear_model/event/strided_slice/stackIlinear/linear_model/linear_model/linear_model/event/strided_slice/stack_1Ilinear/linear_model/linear_model/linear_model/event/strided_slice/stack_2*
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
Clinear/linear_model/linear_model/linear_model/event/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
?
Alinear/linear_model/linear_model/linear_model/event/Reshape/shapePackAlinear/linear_model/linear_model/linear_model/event/strided_sliceClinear/linear_model/linear_model/linear_model/event/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
?
;linear/linear_model/linear_model/linear_model/event/ReshapeReshape;linear/linear_model/linear_model/linear_model/event/ToFloatAlinear/linear_model/linear_model/linear_model/event/Reshape/shape*
T0*
Tshape0*
_output_shapes

: 
?
0linear/linear_model/event/weights/ReadVariableOpReadVariableOp(linear/linear_model/event/weights/part_0*
dtype0*
_output_shapes

:
?
!linear/linear_model/event/weightsIdentity0linear/linear_model/event/weights/ReadVariableOp*
T0*
_output_shapes

:
?
@linear/linear_model/linear_model/linear_model/event/weighted_sumMatMul;linear/linear_model/linear_model/linear_model/event/Reshape!linear/linear_model/event/weights*
T0*
_output_shapes

: *
transpose_a( *
transpose_b( 
?
7linear/linear_model/linear_model/linear_model/prr/ShapeShapeParseExample/ParseExample:5*
T0*
out_type0*
_output_shapes
:
?
Elinear/linear_model/linear_model/linear_model/prr/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Glinear/linear_model/linear_model/linear_model/prr/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Glinear/linear_model/linear_model/linear_model/prr/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
?linear/linear_model/linear_model/linear_model/prr/strided_sliceStridedSlice7linear/linear_model/linear_model/linear_model/prr/ShapeElinear/linear_model/linear_model/linear_model/prr/strided_slice/stackGlinear/linear_model/linear_model/linear_model/prr/strided_slice/stack_1Glinear/linear_model/linear_model/linear_model/prr/strided_slice/stack_2*
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
Alinear/linear_model/linear_model/linear_model/prr/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
?linear/linear_model/linear_model/linear_model/prr/Reshape/shapePack?linear/linear_model/linear_model/linear_model/prr/strided_sliceAlinear/linear_model/linear_model/linear_model/prr/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
?
9linear/linear_model/linear_model/linear_model/prr/ReshapeReshapeParseExample/ParseExample:5?linear/linear_model/linear_model/linear_model/prr/Reshape/shape*
T0*
Tshape0*
_output_shapes

: 
?
.linear/linear_model/prr/weights/ReadVariableOpReadVariableOp&linear/linear_model/prr/weights/part_0*
dtype0*
_output_shapes

:
?
linear/linear_model/prr/weightsIdentity.linear/linear_model/prr/weights/ReadVariableOp*
T0*
_output_shapes

:
?
>linear/linear_model/linear_model/linear_model/prr/weighted_sumMatMul9linear/linear_model/linear_model/linear_model/prr/Reshapelinear/linear_model/prr/weights*
transpose_b( *
T0*
_output_shapes

: *
transpose_a( 
?
Blinear/linear_model/linear_model/linear_model/weighted_sum_no_biasAddNDlinear/linear_model/linear_model/linear_model/drug1_prr/weighted_sumDlinear/linear_model/linear_model/linear_model/drug2_prr/weighted_sumAlinear/linear_model/linear_model/linear_model/drug_1/weighted_sumAlinear/linear_model/linear_model/linear_model/drug_2/weighted_sum@linear/linear_model/linear_model/linear_model/event/weighted_sum>linear/linear_model/linear_model/linear_model/prr/weighted_sum*
N*
_output_shapes

: *
T0
?
/linear/linear_model/bias_weights/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
?
 linear/linear_model/bias_weightsIdentity/linear/linear_model/bias_weights/ReadVariableOp*
T0*
_output_shapes
:
?
:linear/linear_model/linear_model/linear_model/weighted_sumBiasAddBlinear/linear_model/linear_model/linear_model/weighted_sum_no_bias linear/linear_model/bias_weights*
T0*
data_formatNHWC*
_output_shapes

: 
y
linear/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
d
linear/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
linear/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
f
linear/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
linear/strided_sliceStridedSlicelinear/ReadVariableOplinear/strided_slice/stacklinear/strided_slice/stack_1linear/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
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
?
3linear/zero_fraction/total_size/Size/ReadVariableOpReadVariableOp,linear/linear_model/drug1_prr/weights/part_0*
dtype0*
_output_shapes

:
f
$linear/zero_fraction/total_size/SizeConst*
value	B	 R*
dtype0	*
_output_shapes
: 
?
5linear/zero_fraction/total_size/Size_1/ReadVariableOpReadVariableOp,linear/linear_model/drug2_prr/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_1Const*
value	B	 R*
dtype0	*
_output_shapes
: 
?
5linear/zero_fraction/total_size/Size_2/ReadVariableOpReadVariableOp)linear/linear_model/drug_1/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_2Const*
value	B	 R*
dtype0	*
_output_shapes
: 
?
5linear/zero_fraction/total_size/Size_3/ReadVariableOpReadVariableOp)linear/linear_model/drug_2/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_3Const*
dtype0	*
_output_shapes
: *
value	B	 R
?
5linear/zero_fraction/total_size/Size_4/ReadVariableOpReadVariableOp(linear/linear_model/event/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_4Const*
dtype0	*
_output_shapes
: *
value	B	 R
?
5linear/zero_fraction/total_size/Size_5/ReadVariableOpReadVariableOp&linear/linear_model/prr/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_5Const*
dtype0	*
_output_shapes
: *
value	B	 R
?
$linear/zero_fraction/total_size/AddNAddN$linear/zero_fraction/total_size/Size&linear/zero_fraction/total_size/Size_1&linear/zero_fraction/total_size/Size_2&linear/zero_fraction/total_size/Size_3&linear/zero_fraction/total_size/Size_4&linear/zero_fraction/total_size/Size_5*
T0	*
N*
_output_shapes
: 
g
%linear/zero_fraction/total_zero/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
%linear/zero_fraction/total_zero/EqualEqual$linear/zero_fraction/total_size/Size%linear/zero_fraction/total_zero/Const*
_output_shapes
: *
T0	
?
1linear/zero_fraction/total_zero/zero_count/SwitchSwitch%linear/zero_fraction/total_zero/Equal%linear/zero_fraction/total_zero/Equal*
T0
*
_output_shapes
: : 
?
3linear/zero_fraction/total_zero/zero_count/switch_tIdentity3linear/zero_fraction/total_zero/zero_count/Switch:1*
_output_shapes
: *
T0

?
3linear/zero_fraction/total_zero/zero_count/switch_fIdentity1linear/zero_fraction/total_zero/zero_count/Switch*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count/pred_idIdentity%linear/zero_fraction/total_zero/Equal*
T0
*
_output_shapes
: 
?
0linear/zero_fraction/total_zero/zero_count/ConstConst4^linear/zero_fraction/total_zero/zero_count/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpReadVariableOpNlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
?
Nlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/SwitchSwitch,linear/linear_model/drug1_prr/weights/part_02linear/zero_fraction/total_zero/zero_count/pred_id*
T0*?
_class5
31loc:@linear/linear_model/drug1_prr/weights/part_0*
_output_shapes
: : 
?
=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeConst4^linear/zero_fraction/total_zero/zero_count/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/yConst4^linear/zero_fraction/total_zero/zero_count/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Blinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual	LessEqual=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeDlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/SwitchSwitchBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqualBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_tIdentityFlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_fIdentityDlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_idIdentityBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual*
_output_shapes
: *
T0

?
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zerosConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqualNotEqual]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchGlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpElinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*Z
_classP
NLloc:@linear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp
?
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/CastCastTlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

:*

DstT0
?
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/ConstConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
Ylinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_countSumPlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/CastQlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
?
Blinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/CastCastYlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zerosConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchGlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpElinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id*
T0*Z
_classP
NLloc:@linear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/CastCastVlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual*
Truncate( *
_output_shapes

:*

DstT0	*

SrcT0

?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/ConstConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/CastSlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: *
	keep_dims( *

Tidx0
?
Clinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/MergeMerge[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_countBlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	
?
Olinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/subSub=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeClinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/CastCastOlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1Cast=linear/zero_fraction/total_zero/zero_count/zero_fraction/Size*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truedivRealDivPlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/CastRlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
?
Alinear/zero_fraction/total_zero/zero_count/zero_fraction/fractionIdentitySlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count/ToFloatCast9linear/zero_fraction/total_zero/zero_count/ToFloat/Switch*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
9linear/zero_fraction/total_zero/zero_count/ToFloat/SwitchSwitch$linear/zero_fraction/total_size/Size2linear/zero_fraction/total_zero/zero_count/pred_id*
_output_shapes
: : *
T0	*7
_class-
+)loc:@linear/zero_fraction/total_size/Size
?
.linear/zero_fraction/total_zero/zero_count/mulMulAlinear/zero_fraction/total_zero/zero_count/zero_fraction/fraction2linear/zero_fraction/total_zero/zero_count/ToFloat*
T0*
_output_shapes
: 
?
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
?
'linear/zero_fraction/total_zero/Equal_1Equal&linear/zero_fraction/total_size/Size_1'linear/zero_fraction/total_zero/Const_1*
_output_shapes
: *
T0	
?
3linear/zero_fraction/total_zero/zero_count_1/SwitchSwitch'linear/zero_fraction/total_zero/Equal_1'linear/zero_fraction/total_zero/Equal_1*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_1/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_1/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_1/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_1/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_1/pred_idIdentity'linear/zero_fraction/total_zero/Equal_1*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_1/ConstConst6^linear/zero_fraction/total_zero/zero_count_1/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
?
Plinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/SwitchSwitch,linear/linear_model/drug2_prr/weights/part_04linear/zero_fraction/total_zero/zero_count_1/pred_id*
T0*?
_class5
31loc:@linear/linear_model/drug2_prr/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_1/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_1/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual*
_output_shapes
: : *
T0

?
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

?
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

:*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
?
Dlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_count*
Truncate( *
_output_shapes
: *

DstT0	*

SrcT0
?
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

:*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0	
?
Elinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/Size*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
?
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
?
Clinear/zero_fraction/total_zero/zero_count_1/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_1/ToFloatCast;linear/zero_fraction/total_zero/zero_count_1/ToFloat/Switch*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
;linear/zero_fraction/total_zero/zero_count_1/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_14linear/zero_fraction/total_zero/zero_count_1/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_1*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_1/mulMulClinear/zero_fraction/total_zero/zero_count_1/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_1/ToFloat*
_output_shapes
: *
T0
?
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
?
'linear/zero_fraction/total_zero/Equal_2Equal&linear/zero_fraction/total_size/Size_2'linear/zero_fraction/total_zero/Const_2*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_2/SwitchSwitch'linear/zero_fraction/total_zero/Equal_2'linear/zero_fraction/total_zero/Equal_2*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_2/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_2/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_2/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_2/Switch*
_output_shapes
: *
T0

?
4linear/zero_fraction/total_zero/zero_count_2/pred_idIdentity'linear/zero_fraction/total_zero/Equal_2*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_2/ConstConst6^linear/zero_fraction/total_zero/zero_count_2/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
?
Plinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/SwitchSwitch)linear/linear_model/drug_1/weights/part_04linear/zero_fraction/total_zero/zero_count_2/pred_id*
T0*<
_class2
0.loc:@linear/linear_model/drug_1/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_2/switch_f*
dtype0	*
_output_shapes
: *
value	B	 R
?
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_2/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
?
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual*
_output_shapes
: *
T0

?
Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

:*
T0
?
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

:*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
?
Dlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp
?
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

:*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"       
?
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0	
?
Elinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/Size*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_2/ToFloatCast;linear/zero_fraction/total_zero/zero_count_2/ToFloat/Switch*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
?
;linear/zero_fraction/total_zero/zero_count_2/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_24linear/zero_fraction/total_zero/zero_count_2/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_2*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_2/mulMulClinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_2/ToFloat*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_2/MergeMerge0linear/zero_fraction/total_zero/zero_count_2/mul2linear/zero_fraction/total_zero/zero_count_2/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_3Const*
value	B	 R *
dtype0	*
_output_shapes
: 
?
'linear/zero_fraction/total_zero/Equal_3Equal&linear/zero_fraction/total_size/Size_3'linear/zero_fraction/total_zero/Const_3*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_3/SwitchSwitch'linear/zero_fraction/total_zero/Equal_3'linear/zero_fraction/total_zero/Equal_3*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_3/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_3/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_3/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_3/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_3/pred_idIdentity'linear/zero_fraction/total_zero/Equal_3*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_3/ConstConst6^linear/zero_fraction/total_zero/zero_count_3/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
?
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
?
Plinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/SwitchSwitch)linear/linear_model/drug_2/weights/part_04linear/zero_fraction/total_zero/zero_count_3/pred_id*
T0*<
_class2
0.loc:@linear/linear_model/drug_2/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_3/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_3/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual*
_output_shapes
: : *
T0

?
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual*
_output_shapes
: *
T0

?
Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

:*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp
?
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

:*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: *
	keep_dims( *

Tidx0
?
Elinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/Size*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
?
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
?
Clinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_3/ToFloatCast;linear/zero_fraction/total_zero/zero_count_3/ToFloat/Switch*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
;linear/zero_fraction/total_zero/zero_count_3/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_34linear/zero_fraction/total_zero/zero_count_3/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_3*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_3/mulMulClinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_3/ToFloat*
T0*
_output_shapes
: 
?
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
?
'linear/zero_fraction/total_zero/Equal_4Equal&linear/zero_fraction/total_size/Size_4'linear/zero_fraction/total_zero/Const_4*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_4/SwitchSwitch'linear/zero_fraction/total_zero/Equal_4'linear/zero_fraction/total_zero/Equal_4*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_4/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_4/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_4/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_4/Switch*
_output_shapes
: *
T0

?
4linear/zero_fraction/total_zero/zero_count_4/pred_idIdentity'linear/zero_fraction/total_zero/Equal_4*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_4/ConstConst6^linear/zero_fraction/total_zero/zero_count_4/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
?
Plinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/SwitchSwitch(linear/linear_model/event/weights/part_04linear/zero_fraction/total_zero/zero_count_4/pred_id*
T0*;
_class1
/-loc:@linear/linear_model/event/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_4/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_4/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual*
_output_shapes
: : *
T0

?
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

?
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch*
_output_shapes
: *
T0

?
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

:*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
?
Dlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

:*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: *
	keep_dims( *

Tidx0
?
Elinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/sub*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
?
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/Size*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_4/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_4/ToFloatCast;linear/zero_fraction/total_zero/zero_count_4/ToFloat/Switch*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
;linear/zero_fraction/total_zero/zero_count_4/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_44linear/zero_fraction/total_zero/zero_count_4/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_4*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_4/mulMulClinear/zero_fraction/total_zero/zero_count_4/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_4/ToFloat*
T0*
_output_shapes
: 
?
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
?
'linear/zero_fraction/total_zero/Equal_5Equal&linear/zero_fraction/total_size/Size_5'linear/zero_fraction/total_zero/Const_5*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_5/SwitchSwitch'linear/zero_fraction/total_zero/Equal_5'linear/zero_fraction/total_zero/Equal_5*
_output_shapes
: : *
T0

?
5linear/zero_fraction/total_zero/zero_count_5/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_5/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_5/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_5/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_5/pred_idIdentity'linear/zero_fraction/total_zero/Equal_5*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_5/ConstConst6^linear/zero_fraction/total_zero/zero_count_5/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
?
Plinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/SwitchSwitch&linear/linear_model/prr/weights/part_04linear/zero_fraction/total_zero/zero_count_5/pred_id*
T0*9
_class/
-+loc:@linear/linear_model/prr/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_5/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_5/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
?
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch*
_output_shapes
: *
T0

?
Glinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

:*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
?
Dlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp
?
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes

:*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0	
?
Elinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	
?
Qlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
?
Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/Size*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
?
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_5/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
?
4linear/zero_fraction/total_zero/zero_count_5/ToFloatCast;linear/zero_fraction/total_zero/zero_count_5/ToFloat/Switch*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
;linear/zero_fraction/total_zero/zero_count_5/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_54linear/zero_fraction/total_zero/zero_count_5/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_5*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_5/mulMulClinear/zero_fraction/total_zero/zero_count_5/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_5/ToFloat*
_output_shapes
: *
T0
?
2linear/zero_fraction/total_zero/zero_count_5/MergeMerge0linear/zero_fraction/total_zero/zero_count_5/mul2linear/zero_fraction/total_zero/zero_count_5/Const*
N*
_output_shapes
: : *
T0
?
$linear/zero_fraction/total_zero/AddNAddN0linear/zero_fraction/total_zero/zero_count/Merge2linear/zero_fraction/total_zero/zero_count_1/Merge2linear/zero_fraction/total_zero/zero_count_2/Merge2linear/zero_fraction/total_zero/zero_count_3/Merge2linear/zero_fraction/total_zero/zero_count_4/Merge2linear/zero_fraction/total_zero/zero_count_5/Merge*
T0*
N*
_output_shapes
: 
?
)linear/zero_fraction/compute/float32_sizeCast$linear/zero_fraction/total_size/AddN*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
?
$linear/zero_fraction/compute/truedivRealDiv$linear/zero_fraction/total_zero/AddN)linear/zero_fraction/compute/float32_size*
_output_shapes
: *
T0
|
)linear/zero_fraction/zero_fraction_or_nanIdentity$linear/zero_fraction/compute/truediv*
T0*
_output_shapes
: 
?
$linear/fraction_of_zero_weights/tagsConst*0
value'B% Blinear/fraction_of_zero_weights*
dtype0*
_output_shapes
: 
?
linear/fraction_of_zero_weightsScalarSummary$linear/fraction_of_zero_weights/tags)linear/zero_fraction/zero_fraction_or_nan*
T0*
_output_shapes
: 
i
linear/head/logits/ShapeConst*
dtype0*
_output_shapes
:*
valueB"       
n
,linear/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
^
Vlinear/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
O
Glinear/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp

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
|
save/Read/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
X
save/IdentityIdentitysave/Read/ReadVariableOp*
T0*
_output_shapes
:
^
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes
:
?
save/Read_1/ReadVariableOpReadVariableOp,linear/linear_model/drug1_prr/weights/part_0*
dtype0*
_output_shapes

:
`
save/Identity_2Identitysave/Read_1/ReadVariableOp*
T0*
_output_shapes

:
d
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_2/ReadVariableOpReadVariableOp,linear/linear_model/drug2_prr/weights/part_0*
dtype0*
_output_shapes

:
`
save/Identity_4Identitysave/Read_2/ReadVariableOp*
T0*
_output_shapes

:
d
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_3/ReadVariableOpReadVariableOp)linear/linear_model/drug_1/weights/part_0*
dtype0*
_output_shapes

:
`
save/Identity_6Identitysave/Read_3/ReadVariableOp*
T0*
_output_shapes

:
d
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_4/ReadVariableOpReadVariableOp)linear/linear_model/drug_2/weights/part_0*
dtype0*
_output_shapes

:
`
save/Identity_8Identitysave/Read_4/ReadVariableOp*
T0*
_output_shapes

:
d
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
_output_shapes

:*
T0
?
save/Read_5/ReadVariableOpReadVariableOp(linear/linear_model/event/weights/part_0*
dtype0*
_output_shapes

:
a
save/Identity_10Identitysave/Read_5/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_6/ReadVariableOpReadVariableOp&linear/linear_model/prr/weights/part_0*
dtype0*
_output_shapes

:
a
save/Identity_12Identitysave/Read_6/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
T0*
_output_shapes

:
?
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_70243d0039c84e709aab7559d533aa3e/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
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
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
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
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step"/device:CPU:0*
dtypes
2	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
?
save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/Read_7/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
l
save/Identity_14Identitysave/Read_7/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
T0*
_output_shapes
:
?
save/Read_8/ReadVariableOpReadVariableOp,linear/linear_model/drug1_prr/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
p
save/Identity_16Identitysave/Read_8/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_9/ReadVariableOpReadVariableOp,linear/linear_model/drug2_prr/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
p
save/Identity_18Identitysave/Read_9/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_10/ReadVariableOpReadVariableOp)linear/linear_model/drug_1/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_20Identitysave/Read_10/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
f
save/Identity_21Identitysave/Identity_20"/device:CPU:0*
_output_shapes

:*
T0
?
save/Read_11/ReadVariableOpReadVariableOp)linear/linear_model/drug_2/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_22Identitysave/Read_11/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_23Identitysave/Identity_22"/device:CPU:0*
_output_shapes

:*
T0
?
save/Read_12/ReadVariableOpReadVariableOp(linear/linear_model/event/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_24Identitysave/Read_12/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_25Identitysave/Identity_24"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_13/ReadVariableOpReadVariableOp&linear/linear_model/prr/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_26Identitysave/Read_13/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_27Identitysave/Identity_26"/device:CPU:0*
T0*
_output_shapes

:
?
save/SaveV2_1/tensor_namesConst"/device:CPU:0*?
value?B?B linear/linear_model/bias_weightsB%linear/linear_model/drug1_prr/weightsB%linear/linear_model/drug2_prr/weightsB"linear/linear_model/drug_1/weightsB"linear/linear_model/drug_2/weightsB!linear/linear_model/event/weightsBlinear/linear_model/prr/weights*
dtype0*
_output_shapes
:
?
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*h
value_B]B1 0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1*
dtype0*
_output_shapes
:
?
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_15save/Identity_17save/Identity_19save/Identity_21save/Identity_23save/Identity_25save/Identity_27"/device:CPU:0*
dtypes
	2
?
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
_output_shapes
: *
T0*)
_class
loc:@save/ShardedFilename_1
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
save/Identity_28Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
_output_shapes
: *
T0
~
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:* 
valueBBglobal_step
w
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*
_output_shapes
:
?
save/AssignAssignglobal_stepsave/RestoreV2*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
(
save/restore_shardNoOp^save/Assign
?
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*?
value?B?B linear/linear_model/bias_weightsB%linear/linear_model/drug1_prr/weightsB%linear/linear_model/drug2_prr/weightsB"linear/linear_model/drug_1/weightsB"linear/linear_model/drug_2/weightsB!linear/linear_model/event/weightsBlinear/linear_model/prr/weights*
dtype0*
_output_shapes
:
?
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*h
value_B]B1 0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1*
dtype0*
_output_shapes
:
?
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes
	2*V
_output_shapesD
B:::::::
b
save/Identity_29Identitysave/RestoreV2_1"/device:CPU:0*
_output_shapes
:*
T0
?
save/AssignVariableOpAssignVariableOp'linear/linear_model/bias_weights/part_0save/Identity_29"/device:CPU:0*
dtype0
h
save/Identity_30Identitysave/RestoreV2_1:1"/device:CPU:0*
T0*
_output_shapes

:
?
save/AssignVariableOp_1AssignVariableOp,linear/linear_model/drug1_prr/weights/part_0save/Identity_30"/device:CPU:0*
dtype0
h
save/Identity_31Identitysave/RestoreV2_1:2"/device:CPU:0*
T0*
_output_shapes

:
?
save/AssignVariableOp_2AssignVariableOp,linear/linear_model/drug2_prr/weights/part_0save/Identity_31"/device:CPU:0*
dtype0
h
save/Identity_32Identitysave/RestoreV2_1:3"/device:CPU:0*
_output_shapes

:*
T0
?
save/AssignVariableOp_3AssignVariableOp)linear/linear_model/drug_1/weights/part_0save/Identity_32"/device:CPU:0*
dtype0
h
save/Identity_33Identitysave/RestoreV2_1:4"/device:CPU:0*
T0*
_output_shapes

:
?
save/AssignVariableOp_4AssignVariableOp)linear/linear_model/drug_2/weights/part_0save/Identity_33"/device:CPU:0*
dtype0
h
save/Identity_34Identitysave/RestoreV2_1:5"/device:CPU:0*
T0*
_output_shapes

:
?
save/AssignVariableOp_5AssignVariableOp(linear/linear_model/event/weights/part_0save/Identity_34"/device:CPU:0*
dtype0
h
save/Identity_35Identitysave/RestoreV2_1:6"/device:CPU:0*
T0*
_output_shapes

:
?
save/AssignVariableOp_6AssignVariableOp&linear/linear_model/prr/weights/part_0save/Identity_35"/device:CPU:0*
dtype0
?
save/restore_shard_1NoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"?
save/Const:0save/Identity_28:0save/restore_all (5 @F8"A
	summaries4
2
linear/bias:0
!linear/fraction_of_zero_weights:0"?
trainable_variables??
?
.linear/linear_model/drug1_prr/weights/part_0:03linear/linear_model/drug1_prr/weights/part_0/AssignBlinear/linear_model/drug1_prr/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/drug1_prr/weights  "(2@linear/linear_model/drug1_prr/weights/part_0/Initializer/zeros:08
?
.linear/linear_model/drug2_prr/weights/part_0:03linear/linear_model/drug2_prr/weights/part_0/AssignBlinear/linear_model/drug2_prr/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/drug2_prr/weights  "(2@linear/linear_model/drug2_prr/weights/part_0/Initializer/zeros:08
?
+linear/linear_model/drug_1/weights/part_0:00linear/linear_model/drug_1/weights/part_0/Assign?linear/linear_model/drug_1/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/drug_1/weights  "(2=linear/linear_model/drug_1/weights/part_0/Initializer/zeros:08
?
+linear/linear_model/drug_2/weights/part_0:00linear/linear_model/drug_2/weights/part_0/Assign?linear/linear_model/drug_2/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/drug_2/weights  "(2=linear/linear_model/drug_2/weights/part_0/Initializer/zeros:08
?
*linear/linear_model/event/weights/part_0:0/linear/linear_model/event/weights/part_0/Assign>linear/linear_model/event/weights/part_0/Read/ReadVariableOp:0"/
!linear/linear_model/event/weights  "(2<linear/linear_model/event/weights/part_0/Initializer/zeros:08
?
(linear/linear_model/prr/weights/part_0:0-linear/linear_model/prr/weights/part_0/Assign<linear/linear_model/prr/weights/part_0/Read/ReadVariableOp:0"-
linear/linear_model/prr/weights  "(2:linear/linear_model/prr/weights/part_0/Initializer/zeros:08
?
)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08"?
	variables??
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
?
.linear/linear_model/drug1_prr/weights/part_0:03linear/linear_model/drug1_prr/weights/part_0/AssignBlinear/linear_model/drug1_prr/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/drug1_prr/weights  "(2@linear/linear_model/drug1_prr/weights/part_0/Initializer/zeros:08
?
.linear/linear_model/drug2_prr/weights/part_0:03linear/linear_model/drug2_prr/weights/part_0/AssignBlinear/linear_model/drug2_prr/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/drug2_prr/weights  "(2@linear/linear_model/drug2_prr/weights/part_0/Initializer/zeros:08
?
+linear/linear_model/drug_1/weights/part_0:00linear/linear_model/drug_1/weights/part_0/Assign?linear/linear_model/drug_1/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/drug_1/weights  "(2=linear/linear_model/drug_1/weights/part_0/Initializer/zeros:08
?
+linear/linear_model/drug_2/weights/part_0:00linear/linear_model/drug_2/weights/part_0/Assign?linear/linear_model/drug_2/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/drug_2/weights  "(2=linear/linear_model/drug_2/weights/part_0/Initializer/zeros:08
?
*linear/linear_model/event/weights/part_0:0/linear/linear_model/event/weights/part_0/Assign>linear/linear_model/event/weights/part_0/Read/ReadVariableOp:0"/
!linear/linear_model/event/weights  "(2<linear/linear_model/event/weights/part_0/Initializer/zeros:08
?
(linear/linear_model/prr/weights/part_0:0-linear/linear_model/prr/weights/part_0/Assign<linear/linear_model/prr/weights/part_0/Read/ReadVariableOp:0"-
linear/linear_model/prr/weights  "(2:linear/linear_model/prr/weights/part_0/Initializer/zeros:08
?
)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"??
cond_context????
?
4linear/zero_fraction/total_zero/zero_count/cond_text4linear/zero_fraction/total_zero/zero_count/pred_id:05linear/zero_fraction/total_zero/zero_count/switch_t:0 *?
2linear/zero_fraction/total_zero/zero_count/Const:0
4linear/zero_fraction/total_zero/zero_count/pred_id:0
5linear/zero_fraction/total_zero/zero_count/switch_t:0l
4linear/zero_fraction/total_zero/zero_count/pred_id:04linear/zero_fraction/total_zero/zero_count/pred_id:0
?.
6linear/zero_fraction/total_zero/zero_count/cond_text_14linear/zero_fraction/total_zero/zero_count/pred_id:05linear/zero_fraction/total_zero/zero_count/switch_f:0*?
.linear/linear_model/drug1_prr/weights/part_0:0
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
Clinear/zero_fraction/total_zero/zero_count/zero_fraction/fraction:0l
4linear/zero_fraction/total_zero/zero_count/pred_id:04linear/zero_fraction/total_zero/zero_count/pred_id:0e
&linear/zero_fraction/total_size/Size:0;linear/zero_fraction/total_zero/zero_count/ToFloat/Switch:0?
.linear/linear_model/drug1_prr/weights/part_0:0Plinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch:02?

?

Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/cond_textGlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0 *?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast:0
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Cast:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual:0
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:02?

?

Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/cond_text_1Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0*?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Cast:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const:0
_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Xlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_1/cond_text6linear/zero_fraction/total_zero/zero_count_1/pred_id:07linear/zero_fraction/total_zero/zero_count_1/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_1/Const:0
6linear/zero_fraction/total_zero/zero_count_1/pred_id:0
7linear/zero_fraction/total_zero/zero_count_1/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_1/pred_id:06linear/zero_fraction/total_zero/zero_count_1/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_1/cond_text_16linear/zero_fraction/total_zero/zero_count_1/pred_id:07linear/zero_fraction/total_zero/zero_count_1/switch_f:0*?
.linear/linear_model/drug2_prr/weights/part_0:0
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
(linear/zero_fraction/total_size/Size_1:0=linear/zero_fraction/total_zero/zero_count_1/ToFloat/Switch:0p
6linear/zero_fraction/total_zero/zero_count_1/pred_id:06linear/zero_fraction/total_zero/zero_count_1/pred_id:0?
.linear/linear_model/drug2_prr/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_2/cond_text6linear/zero_fraction/total_zero/zero_count_2/pred_id:07linear/zero_fraction/total_zero/zero_count_2/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_2/Const:0
6linear/zero_fraction/total_zero/zero_count_2/pred_id:0
7linear/zero_fraction/total_zero/zero_count_2/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_2/pred_id:06linear/zero_fraction/total_zero/zero_count_2/pred_id:0
?/
8linear/zero_fraction/total_zero/zero_count_2/cond_text_16linear/zero_fraction/total_zero/zero_count_2/pred_id:07linear/zero_fraction/total_zero/zero_count_2/switch_f:0*?
+linear/linear_model/drug_1/weights/part_0:0
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
Elinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fraction:0?
+linear/linear_model/drug_1/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/Switch:0p
6linear/zero_fraction/total_zero/zero_count_2/pred_id:06linear/zero_fraction/total_zero/zero_count_2/pred_id:0i
(linear/zero_fraction/total_size/Size_2:0=linear/zero_fraction/total_zero/zero_count_2/ToFloat/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t:0?
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0?
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:12?

?

Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_3/cond_text6linear/zero_fraction/total_zero/zero_count_3/pred_id:07linear/zero_fraction/total_zero/zero_count_3/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_3/Const:0
6linear/zero_fraction/total_zero/zero_count_3/pred_id:0
7linear/zero_fraction/total_zero/zero_count_3/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_3/pred_id:06linear/zero_fraction/total_zero/zero_count_3/pred_id:0
?/
8linear/zero_fraction/total_zero/zero_count_3/cond_text_16linear/zero_fraction/total_zero/zero_count_3/pred_id:07linear/zero_fraction/total_zero/zero_count_3/switch_f:0*?
+linear/linear_model/drug_2/weights/part_0:0
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
Elinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_3/pred_id:06linear/zero_fraction/total_zero/zero_count_3/pred_id:0?
+linear/linear_model/drug_2/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/Switch:0i
(linear/zero_fraction/total_size/Size_3:0=linear/zero_fraction/total_zero/zero_count_3/ToFloat/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_4/cond_text6linear/zero_fraction/total_zero/zero_count_4/pred_id:07linear/zero_fraction/total_zero/zero_count_4/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_4/Const:0
6linear/zero_fraction/total_zero/zero_count_4/pred_id:0
7linear/zero_fraction/total_zero/zero_count_4/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_4/pred_id:06linear/zero_fraction/total_zero/zero_count_4/pred_id:0
?/
8linear/zero_fraction/total_zero/zero_count_4/cond_text_16linear/zero_fraction/total_zero/zero_count_4/pred_id:07linear/zero_fraction/total_zero/zero_count_4/switch_f:0*?
*linear/linear_model/event/weights/part_0:0
(linear/zero_fraction/total_size/Size_4:0
=linear/zero_fraction/total_zero/zero_count_4/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_4/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_4/mul:0
6linear/zero_fraction/total_zero/zero_count_4/pred_id:0
7linear/zero_fraction/total_zero/zero_count_4/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
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
6linear/zero_fraction/total_zero/zero_count_4/pred_id:06linear/zero_fraction/total_zero/zero_count_4/pred_id:0?
*linear/linear_model/event/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/Switch:0i
(linear/zero_fraction/total_size/Size_4:0=linear/zero_fraction/total_zero/zero_count_4/ToFloat/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_5/cond_text6linear/zero_fraction/total_zero/zero_count_5/pred_id:07linear/zero_fraction/total_zero/zero_count_5/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_5/Const:0
6linear/zero_fraction/total_zero/zero_count_5/pred_id:0
7linear/zero_fraction/total_zero/zero_count_5/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_5/pred_id:06linear/zero_fraction/total_zero/zero_count_5/pred_id:0
?/
8linear/zero_fraction/total_zero/zero_count_5/cond_text_16linear/zero_fraction/total_zero/zero_count_5/pred_id:07linear/zero_fraction/total_zero/zero_count_5/switch_f:0*?
(linear/linear_model/prr/weights/part_0:0
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
6linear/zero_fraction/total_zero/zero_count_5/pred_id:06linear/zero_fraction/total_zero/zero_count_5/pred_id:0i
(linear/zero_fraction/total_size/Size_5:0=linear/zero_fraction/total_zero/zero_count_5/ToFloat/Switch:0~
(linear/linear_model/prr/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0"%
saved_model_main_op


group_deps*?
serving_default?
(
inputs
input_example_tensor:0S
outputsH
<linear/linear_model/linear_model/linear_model/weighted_sum:0 tensorflow/serving/regress*?
predict?
*
examples
input_example_tensor:0W
predictionsH
<linear/linear_model/linear_model/linear_model/weighted_sum:0 tensorflow/serving/predict*?

regression?
(
inputs
input_example_tensor:0S
outputsH
<linear/linear_model/linear_model/linear_model/weighted_sum:0 tensorflow/serving/regress