# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv, erf, exp, tanh
from sys.info import num_physical_cores, simdwidthof

from algorithm import elementwise
from buffer import NDBuffer
from memory import UnsafePointer

from utils.index import IndexList


# CHECK-LABEL: test_elementwise_1d
def test_elementwise_1d():
    print("== test_elementwise_1d")

    alias num_elements = 2
    var ptr = UnsafePointer[Float32].alloc(num_elements)

    var vector = NDBuffer[DType.float32, 1, _, num_elements](ptr)
    var ptr2 = UnsafePointer[Float32].alloc(num_elements)
    var vector2 = NDBuffer[DType.float32, 1, _, num_elements](ptr2)

    for i in range(len(vector)):
        vector[i] = i
        vector2[i] = i + len(vector)  # or any constant/vector you want to dot with
    var sum: Float32 = 0.0


    @always_inline
    # @__copy_capture(vector, vector2, sum)
    @parameter
    fn func[simd_width: Int, rank: Int](idx: IndexList[rank]):
        var elem = vector.load[width=simd_width](idx[0])
        var other = vector2.load[width=simd_width](idx[0])
        # var val = elem 
        for i in range(simd_width):
            sum += elem[i] * other[i]

    elementwise[func, simdwidthof[DType.float32]()](IndexList[1](num_elements))

    # CHECK: 2.051446{{[0-9]+}}
    print("Dot Product:", sum)

    ptr.free()


def main():
    test_elementwise_1d()