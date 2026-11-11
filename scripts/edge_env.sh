#!/usr/bin/env bash
# 用法（在项目根目录）：source scripts/edge_env.sh
# 作用：把虚拟环境里随pip安装的NVIDIA运行库目录加入LD_LIBRARY_PATH，
#       让onnxruntime的CUDA/TensorRT提供器能找到 libcurand、libcudnn、libnvinfer 等库。
# 注意：必须在启动Python之前设置(系统加载器在进程启动时读取该变量)，所以要先source再uv run。

SP=$(ls -d "$PWD"/.venv/lib/python*/site-packages | head -1)    # 虚拟环境的site-packages目录
LIBS=""
# 1) nvidia/ 下所有名为lib的目录(cuDNN、cuBLAS、cuRAND等)
for d in $(find "$SP/nvidia" -type d -name lib 2>/dev/null); do LIBS="$LIBS:$d"; done
# 2) TensorRT的库目录(如果安装了tensorrt，库文件直接放在tensorrt_libs里)
[ -d "$SP/tensorrt_libs" ] && LIBS="$LIBS:$SP/tensorrt_libs"

export LD_LIBRARY_PATH="${LIBS#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH now has $(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -c .) entries"
