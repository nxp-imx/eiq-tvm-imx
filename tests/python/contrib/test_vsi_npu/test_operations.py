# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
from tvm import relay
from tvm.relay import testing
import numpy as np
from infrastructure import verify_vsi_result
import argparse

parser = argparse.ArgumentParser(description='VSI-NPU test script for tflite models.')
parser.add_argument('-i', '--ip', type=str, required=True,
                    help='ip address for remote target board')
parser.add_argument('-p', '--port', type=int, default=9090,
                    help='port number for remote target board')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu instead of npu or gpu')
args = parser.parse_args()

def verify_op_result(mod, params, data_shape, out_shape, dtype):
    return verify_vsi_result(mod, params, data_shape, out_shape, dtype, args.ip, args.port, args.cpu)

def _single_operation_test(relay_nn_func, dtype, data_shape, out_shape,
        *args, **kargs):
    op_name = relay_nn_func.__name__.upper()
    print("Testing {0: <50}".format(op_name), end="")
    data = relay.var("data", shape=data_shape, dtype=dtype)

    out = relay_nn_func(data, *args, **kargs)

    args = relay.analysis.free_vars(out)
    net = relay.Function(args, out)

    mod, params = relay.testing.init.create_workload(net)

    verify_op_result(mod, params, data_shape, out_shape, dtype)

def test_global_avg_pool2d():
    func = relay.nn.global_avg_pool2d

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = (1, 20, 1, 1)
    _single_operation_test(func, dtype, data_shape, out_shape)


def test_global_max_pool2d():
    func = relay.nn.global_max_pool2d

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = (1, 20, 1, 1)
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_avg_pool2d():
    func = relay.nn.avg_pool2d

    dtype = "float32"
    data_shape = (1, 1, 20, 20)
    out_shape = (1, 1, 10, 10)
    _single_operation_test(func, dtype, data_shape, out_shape, pool_size=(3, 3),
            strides=(2, 2), padding=(1,1,1,1))

def test_max_pool2d():
    func = relay.nn.max_pool2d

    dtype = "float32"
    data_shape = (1, 1, 20, 20)
    out_shape = (1, 1, 10, 10)
    _single_operation_test(func, dtype, data_shape, out_shape, pool_size=(2, 2),
            strides=(2, 2), padding=(0,0,0,0))

def test_softmax():
    func = relay.nn.softmax

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = data_shape
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_relu():
    func = relay.nn.relu

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = data_shape
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_add():
    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = data_shape

    def get_workload(data_shape, dtype="float32"):
        '''customized keywords(like data0,data1...) are not supported in \
        relay.testing.init.create_workload
        '''
        data0 = relay.var("data", shape=data_shape, dtype=dtype)
        data1 = relay.var("weight", shape=data_shape, dtype=dtype)

        out = relay.add(data0, data1)

        args = relay.analysis.free_vars(out)
        net = relay.Function(args, out)

        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("ADD"), end="")
    mod, params = get_workload(data_shape, dtype)
    verify_op_result(mod, params, data_shape, out_shape, dtype)


def test_batch_flatten():
    func = relay.nn.batch_flatten

    dtype = "float32"
    data_shape = (1, 5, 10, 10)
    out_shape = (1, 500)
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_batch_norm():
    data_shape = (1, 4)
    c_shape = (4,)
    out_shape = (1, 4)
    dtype="float32"

    def get_workload(data_shape, weight_shape, dtype="float32"):
        data = relay.var("data", shape=data_shape, dtype=dtype)

        w = tvm.nd.array(np.ones(weight_shape, dtype))
        gamma = relay.const(w, dtype)
        beta = relay.const(w, dtype)
        moving_mean = relay.const(w, dtype)
        moving_var = relay.const(w, dtype)

        bn = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var)
        out = bn[0]

        args = relay.analysis.free_vars(out)
        net = relay.Function(args, out)

        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("BATCH_NORM"), end="")
    mod, params = get_workload(data_shape, c_shape, dtype)
    verify_op_result(mod, params, data_shape, out_shape, dtype)

def test_conv2d():
    data_shape = (1, 256, 64, 64)
    weight_shape = (256, 256, 3, 3)
    out_shape = (1, 256, 64, 64)
    dtype="float32"
    Pad=(1,1,1,1)
    Strides=(1,1)
    Dilation=(1,1)
    Ksize=(3,3)
    Groups=1

    def get_workload(data_shape, weight_shape, dtype="float32"):
        """Function to construct a MobileNet"""
        data = relay.var("data", shape=data_shape, dtype=dtype)
        weight = relay.var("conv_weight")
        conv = relay.nn.conv2d(
            data,
            weight,
            channels=weight_shape[0],
            kernel_size=Ksize,
            strides=Strides,
            padding=Pad,
            groups=Groups,
            data_layout="NCHW",
            kernel_layout="OIHW"
        )
        args = relay.analysis.free_vars(conv)
        net = relay.Function(args, conv)
        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("CONV2D"), end="")
    mod, params = get_workload(data_shape, weight_shape, dtype)
    verify_op_result(mod, params, data_shape, out_shape, dtype)


def test_dense():
    data_shape = (1, 784)
    weight_shape = (128, 784)
    out_shape = (1, 128)
    dtype="float32"

    def get_workload(data_shape, weight_shape, dtype="float32"):
        data = relay.var("data", shape=data_shape, dtype=dtype)
        fc1 = relay.nn.dense(data, relay.var("fc1_weight"), units=weight_shape[0])
        fc = relay.nn.bias_add(fc1, relay.var("fc1_bias"), axis=1)
        args = relay.analysis.free_vars(fc)
        net = relay.Function(args, fc)

        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("DENSE_BIAS_ADD"), end="")
    mod, params = get_workload(data_shape, weight_shape, dtype)
    verify_op_result(mod, params, data_shape, out_shape, dtype)

def test_concatenate():
    dtype = "float32"
    data_shape = (1, 20, 12)
    out_shape = (1, 20, 24)

    def get_workload(data_shape, dtype="float32"):
        '''customized keywords(like data0,data1...) are not supported in \
        relay.testing.init.create_workload
        '''
        data0 = relay.var("data", shape=data_shape, dtype=dtype)
        data1 = relay.var("bias", shape=data_shape, dtype=dtype)

        out = relay.concatenate([data0, data1], axis=-1)
        args = relay.analysis.free_vars(out)
        net = relay.Function(args, out)

        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("CONCATENATE"), end="")
    mod, params = get_workload(data_shape, dtype)
    verify_op_result(mod, params, data_shape, out_shape, dtype)

def test_image_resize():
    data_dtype = 'float32'
    in_size = (33, 33)
    out_size = (257, 257)
    channel = (21,)

    def run_test(layout, method, mode):
        if (layout == "NHWC"):
            data_shape = (1,) + in_size + channel
            out_shape = (1,) + out_size + channel
        else:
            data_shape = (1,) + channel + in_size
            out_shape = (1,) + channel + out_size
        x = relay.var("data", shape=data_shape, dtype=data_dtype)
        out = relay.image.resize(x, out_size, layout, method, mode)
        args = relay.analysis.free_vars(out)
        net = relay.Function(args, out)

        mod, params = relay.testing.init.create_workload(net)
        verify_op_result(mod, params, data_shape, out_shape, data_dtype)
    print("Testing {0: <50}".format("RESIZE_1"), end="")
    run_test("NHWC", "nearest_neighbor", "asymmetric")
    print("Testing {0: <50}".format("RESIZE_2"), end="")
    run_test("NHWC", "bilinear", "asymmetric")
    print("Testing {0: <50}".format("RESIZE_3"), end="")
    run_test("NHWC", "bilinear", "half_pixel")
    print("Testing {0: <50}".format("RESIZE_4"), end="")
    run_test("NHWC", "bilinear", "align_corners")
    print("Testing {0: <50}".format("RESIZE_5"), end="")
    run_test("NCHW", "nearest_neighbor", "asymmetric")
    print("Testing {0: <50}".format("RESIZE_6"), end="")
    run_test("NCHW", "bilinear", "asymmetric")
    print("Testing {0: <50}".format("RESIZE_7"), end="")
    run_test("NCHW", "bilinear", "half_pixel")
    print("Testing {0: <50}".format("RESIZE_8"), end="")
    run_test("NCHW", "bilinear", "align_corners")

def test_dropout():
    func = relay.nn.dropout

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = data_shape
    _single_operation_test(func, dtype, data_shape, out_shape, rate=0.5)

def test_qnn_add():
    data_dtype = "uint8"
    data_shape = (1, 20, 4, 8)
    out_shape = (1, 20, 4, 8)

    x = relay.var("data", shape=data_shape, dtype=data_dtype)
    y = relay.var("weight", shape=data_shape, dtype=data_dtype)
    out = relay.qnn.op.add(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.00784314, "float32"),
        lhs_zero_point=relay.const(127, "int32"),
        rhs_scale=relay.const(0.00784314, "float32"),
        rhs_zero_point=relay.const(127, "int32"),
        output_scale=relay.const(0.00784314, "float32"),
        output_zero_point=relay.const(127, "int32"),
    )
    args = relay.analysis.free_vars(out)
    net = relay.Function(args, out)

    print("Testing {0: <50}".format("QNN.ADD"), end="")
    mod, params = relay.testing.init.create_workload(net)
    verify_op_result(mod, params, data_shape, out_shape, data_dtype)


if __name__ == "__main__":
    test_qnn_add()
    test_batch_norm()
    test_softmax()
    test_add()
    test_relu()
    test_batch_flatten()
    test_avg_pool2d()
    test_max_pool2d()
    test_global_avg_pool2d()
    test_global_max_pool2d()
    test_conv2d()
    test_dense()
    test_dropout()
    test_concatenate()
    test_image_resize()

