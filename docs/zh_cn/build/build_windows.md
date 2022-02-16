- [Windows 下构建方式](#windows-下构建方式)
  - [源码安装](#源码安装)
    - [安装构建编译工具链](#安装构建编译工具链)
    - [安装依赖包](#安装依赖包)
      - [安装 MMDeploy Converter 依赖](#安装-mmdeploy-converter-依赖)
      - [安装 MMDeploy SDK 依赖](#安装-mmdeploy-sdk-依赖)
      - [安装推理引擎](#安装推理引擎)
    - [编译 MMDeploy](#编译-mmdeploy)
      - [编译安装 Model Converter](#编译安装-model-converter)
      - [编译安装 SDK](#编译安装-sdk)
    - [注意事项](#注意事项)
# Windows 下构建方式

目前，MMDeploy 在 Windows 平台下仅提供源码编译安装方式。将来，会提供预编译包方式。

## 源码安装

### 安装构建编译工具链
visual studio 2019

### 安装依赖包
  
#### 安装 MMDeploy Converter 依赖
<table>
<thead>
  <tr>
    <th>名称 </th>
    <th>安装方法 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>conda </td>
    <td>强烈建议安装conda，或者miniconda。比如， <br>https://repo.anaconda.com/miniconda/Miniconda3-py37_4.11.0-Windows-x86_64.exe <br>安装完毕后，打开系统开始菜单，输入prompt，选择并打开 anaconda powershell prompt。 <br><b>下文中的安装命令均是在powershell中经过测试验证。</b> </td>
  </tr>
  <tr>
    <td>pytorch </td>
    <td>
    参考<a href="https://pytorch.org/get-started/locally/">pytorch官网</a>，根据系统环境, 选择合适的预编译包进行安装。比如, <br>
    1. pytorch1.8 + cuda111 环境 <br>
    <pre><code>
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    </code></pre>
    1. pytorch1.8 + cpu 环境 <br>
    <pre><code>
    pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    </code></pre>
    </td>
  </tr>
  <tr>
    <td>mmcv-full </td>
    <td>参考<a href="https://github.com/open-mmlab/mmcv">mmcv官网</a>，根据系统环境，选择预编译包进行安装。比如，<br>
    1. pytorch1.8 + cu111 + mmcv1.4.0
    <pre><code>
    pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
    </code></pre>
    1. pytorch1.8 + cpu + mmcv1.4.0
    <pre><code>
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html
    </code></pre>
    </td>
  </tr>
</tbody>
</table>


#### 安装 MMDeploy SDK 依赖
<table>
<thead>
  <tr>
    <th>名称 </th>
    <th>安装方法 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>spdlog </td>
    <td>spdlog是一个精巧的日志管理库。请参考如下命令安装： <br>
    1. 下载 https://github.com/gabime/spdlog/archive/refs/tags/v1.9.2.zip <br>
    2. 解压后，进入到文件夹 spdlog-v1.9.2 <br>
    3. 执行编译安装命令 <br>
    <pre><code>
    mkdir build
    cd build
    cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j --config Release 
    make --build . --target install -j --config Release
    </code></pre>
   </td>
  </tr>
  <tr>
    <td>OpenCV </td>
    <td>下载并安装 OpenCV 在 windows 下的预编译包: https://github.com/opencv/opencv/releases/download/4.5.5/opencv-4.5.5-vc14_vc15.exe</td>
  </tr>
  <tr>
    <td>pplcv </td>
    <td>pplcv 是在x86和cuda平台下的高性能图像处理库。 <b>此依赖项为可选项，只有在cuda平台下，才需安装</b><br>
    1. 下载 https://github.com/openppl-public/ppl.cv/archive/refs/tags/v0.6.2.zip
    2. 解压后，进入文件夹 ppl.cv-0.6.2 <br>
    3. 执行编译安装命令 <br>
    <pre><code>
    ./build.bat -G "Visual Studio 16 2019" -T v142 -A x64 -DHPCC_USE_CUDA=ON -DHPCC_MSVC_MD=ON
    </code></pre>
   </td>
  </tr>
</tbody>
</table>



#### 安装推理引擎
<table>
<thead>
  <tr>
    <th>推理引擎 </th>
    <th>依赖包</th>
    <th>安装方法 </th>
  </tr>
</thead>
<tbody>
    <tr>
    <td>ONNXRuntime</td>
    <td>onnxruntime </td>
    <td> </td>
  </tr>
  <tr>
    <td rowspan="2">TensorRT<br> </td>
    <td>TensorRT <br> </td>
    <td> 
    1. 从NVIDIA官网下载二进制包, 比如，<br>
   https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.1/zip/tensorrt-8.2.1.8.windows10.x86_64.cuda-11.4.cudnn8.2.zip <br>
    1. 解压二进制包 <br>
    2. 安装 tensorrt 的 python package <br>
   <pre><code>
   pip install 
   </code></pre>
   </td>
  </tr>
  <tr>
    <td>cudnn </td>
    <td>
    1. 从NVIDIA官网下载二进制包, 比如, <br>
   https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-windows-x64-v8.2.1.32.zip <br>
    2. 解压二进制包 <br>
   </td>
  </tr>
  <tr>
    <td>PPL.NN</td>
    <td>ppl.nn </td>
    <td> TODO </td>
  </tr>
  <tr>
    <td>OpenVINO</td>
    <td>openvino </td>
    <td>TODO </td>
  </tr>
  <tr>
    <td>ncnn </td>
    <td>ncnn </td>
    <td>TODO </td>
  </tr>
</tbody>
</table>

### 编译 MMDeploy

#### 编译安装 Model Converter

#### 编译安装 SDK
- 设置目标设备

  cpu 是 SDK 目标设备的默认选项。你也可以通过`MMDEPLOY_TARGET_DEVICES`传入其他设备名称。当有多个设备时，设备名称之间使用分号隔开。
  比如，`-DMMDEPLOY_TARGET_DEVICES="cpu;cuda"`。
  当前，SDK支持以下设备，

  | 设备 | 名称 | 查找路径                           |
  | :--- | :--- | :--------------------------------- |
  | Host | cpu  | N/A                                |
  | CUDA | cuda | CUDA_TOOLKIT_ROOT_DIR 和 pplcv_DIR |

  如果你的开发环境中有多个cuda版本，则需要通过`-DCUDA_TOOLKIT_ROOT_DIR=/path/of/cuda`来明确使用的版本。
  于此同时，还需设置`-Dpplcv_DIR=ppl.cv/path/install/lib/cmake/ppl`，用以编译cuda平台下的图像处理算子。

- 设置推理后端

  **默认情况下，SDK不设置任何后端**, 因为它与应用场景高度相关。你可以通过设置`MMDEPLOY_TARGET_BACKENDS`激活感兴趣的推理后端。
  当选择多个时， 中间使用分号隔开。比如，`-DMMDEPLOY_TARGET_BACKENDS="trt;ort;pplnn;ncnn;openvino"`
  构建时，几乎每个后端，都需设置一些环境变量，用来查找依赖包。
  下表展示了目前SDK支持的后端，以及构建时，每个后端需要设置的变量。

  | 推理引擎    | 名称     | 查找路径                 |
  | :---------- | :------- | :----------------------- |
  | PPL.nn      | pplnn    | pplnn_DIR                |
  | ncnn        | ncnn     | ncnn_DIR                 |
  | ONNXRuntime | ort      | ONNXRUNTIME_DIR          |
  | TensorRT    | trt      | TENSORRT_DIR & CUDNN_DIR |
  | OpenVINO    | openvino | InferenceEngine_DIR      |

- 设置后处理组件

  需要通过`MMDEPLOY_CODEBASES`设置SDK后处理组件，才能加载OpenMMLab算法仓库的后处理功能。已支持的算法仓库有'mmcls'，'mmdet'，'mmedit'，'mmseg'和'mmocr'。
  如果选择多个codebase，中间使用分号隔开。比如，`-DMMDEPLOY_CODEBASES=mmcls;mmdet`。也可以通过`-DMMDEPLOY_CODEBASES=all`方式，加载所有codebase。

- 汇总以上

  下文展示2个构建SDK的样例，分别用于不同的运行环境。
  使用cpu设备和ONNXRuntime推理，请参考

  ```PowerShell
  mkdir build
  cd build
  cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DONNXRUNTIME_DIR=%path%to%onnxruntime% \
      -DMMDEPLOY_TARGET_DEVICES=cpu \
      -DMMDEPLOY_TARGET_BACKENDS=ort \
      -DMMDEPLOY_CODEBASES=all \
      -Dspdlog_DIR=%spdlog_dir%/build/install/lib/cmake/spdlog \
      -DOpenCV_DIR=%opencv_dir%
  cmake --build . -- -j --config Release
  cmake --install . --config Release
  ```

  使用cuda设备和TensorRT推理，请按照此例构建

  ```PowerShell
   mkdir build && cd build
   cmake .. \
     -DMMDEPLOY_BUILD_SDK=ON \
     -DCMAKE_CXX_COMPILER=g++-7 \
     -Dpplcv_DIR=/path/to/ppl.cv/cuda-build/install/lib/cmake/ppl \
     -DTENSORRT_DIR=/path/to/tensorrt \
     -DCUDNN_DIR=/path/to/cudnn \
     -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
     -DMMDEPLOY_TARGET_BACKENDS=trt \
     -DMMDEPLOY_CODEBASES=all
   cmake --build . -- -j$(nproc) && cmake --install .
  ```

### 注意事项
  1. Release / Debug 库不能混用。MMDeploy要是编译Debug版本，所有第三方依赖都要是Debug版本。