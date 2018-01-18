# tflite_zynq
Example code and instructions on getting Tensorflow Lite running on a Xilinx Zynq

The first step is building a compiler for the Zynq and getting a Linux system up and running.
[Xilinx](http://www.wiki.xilinx.com/) has some good resources on how to do that. Although building
a custom toolchain and development system with [Buildroot](https://buildroot.org/) is also fairly straightforward.

# Buildroot

Download buildroot

```
git clone git://git.busybox.net/buildroot
cd buildroot
make zynq_zed_defconfig
make nconfig
```

This will bring up a window that allows you to configure buildroot. You can now configure buildroot for your needs. I've checked in my buildroot config located in the **buildroot** directory. Just copy the *config* file to *.config* in the buildroot directory.

While still in the buildroot directory type the following

```
make -jN
```

Where N is the number of jobs, generally the number of cores you have. Now go to lunch or bed or something. When you get back Buildroot will have created the **output** directory. If you look in the output directory you will find the following directories:
build
host
images
staging
target

The **host** directory will contain your compiler, libraries, etc. **images** contains the Xilinx fsbl (first level bootloader), u-boot, the rootfs, and the device tree blobs. It doesn't contain an FPGA image. So you may want to grab a prebuild one and put it on the SD card, otherwise u-boot will fail to boot.

# Build Tensorflow Lite

This takes a bit of gynmastics to get working. I thought it should be straightforward, but I was unfamilar with Bazel and the intricacies.

First checkout [Tensorflow](https://github.com/tensorflow/tensorflow).
The easy method is to copy the tensorflow directory from this repo into your tensorflow repo. Otherwise follow along below

I copied the iOS Simple example and created a command line stand alone program. It is located in the *tensorflow/tensorflow/contrib/lite/examples/simplelite* directory. It uses OpenCV to load and resize the image before feeding it into the TFLITE.

I took the *CROSSTOOL.tpl* and made a copy of it. It is located in the *tensorflow/third_party/toolchains/cpus/arm/* directory. I couldn't figure out how to get the %{ARM_COMPILER_PATH}% to point to the buildroot compilers. I'm sure there is a more elegant approach, but I simple replaced %{ARM_COMPILER_PATH}% with the hardcoded path to the buildroot compiler.

```
tool_path { name: "ar" path: "/home/nlbutts/projects/buildroot/output/host/bin/arm-buildroot-linux-gnueabihf-ar" }
```

I then updated the *cxx_builtin_include_directory* and some of the compiler flags to force the floating point ABI to use *hard*.

I did have to add the following line to the BUILD file in the *tensorflow/third_party/toolchains/cpus/arm/* directory. This seems like an issue that needs to be fixed.
licenses(["notice"])  # Apache 2.0

The last change I needed to make was to TFLITE's floating point ABI. In *tensorflow/tensorflow/contrib/lite/kernels/internal/BUILD* they override compiler flags based on the target CPU.

```
    ":armv7a": [
        "-O3",
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ],
```

Android and iOS must use softfp. But in my experience you pay a slight performance penality. Therefore I compiled my Buildroot system to use -mfloat-abi=hard. Therefore this line needs to be changed to:
```
    ":armv7a": [
        "-O3",
        "-mfpu=neon",
        "-mfloat-abi=hard",
    ],
```

# Cross Compiling TFLITE and Running

Now we can cross compile TFLITE for the zynq. Navigate to your Tensorflow directory.
```
bazel build --crosstool_top=third_party/toolchains/cpus/arm:toolchain --cpu=armv7a tensorflow/contrib/lite/examples/simplelite:simplelite
```

If everything worked well you should have a statically linked simplelite file in the *bazel-bin/tensorflow/contrib/lite/examples/simplelite*

Copy that to your Zynq target long with the files in the *zynq_target_files* directory. Log into your Zynq target and run it with the following commands and you will get the response shown below. I don't know why I get the link error with libneuralnetworks.so. Although it still works.

```
./simplelite mobilenet.lite labels.txt grace_hopper.jpg
Using graph mobilenet.lite with labels labels.txt and image grace_hopper.jpg
nnapi error: unable to open library libneuralnetworks.so
Loaded model mobilenet.lite
resolved reporter
Getting the input tensor
Resizing input tensor
input image size: 517x606x3
output image size: 224x224x3
Requesting output node
TFLite took 2855029us.
Predictions: 653 0.797  military uniform
```

As you can see it takes about 2.8 seconds to run the [MobileNet_v1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md). Specifically I used the MobileNet_v1_1.0_224, which should take 569 million MACs per inference. The Zynq is running at 666 MHz. On an Intel i7-6700HQ, this inference takes ~150 ms.