![title](https://github.com/DylanWaken/DylannDocs/blob/master/assets/Name.png)
# THE DYLANN PROJECT

----

## What is Dylann?

The Dynamic Lightweight Framework for Algorithmic Neural Networks (Dy.L.A.N.N) is for designing and training neural nets in C++ and CUDA environment. The project is intended to be
highly modular and easy to use, especially for the application in game engines or other non-python contexts
that seeks direct access to nn models. The project is still in its early stages, but the core architecture
is basically shaped and ready for applications. 

For now, Dylann is only accessable through CUDA C as a runtime application. When the basic instruction sets are completed 
the base code would be compiled into libraries that can be linked in other projects. Dylann will not be including external
libraries else than CUDA stl, OpenCV (and audio libraries in the future).

![menu-title](https://github.com/DylanWaken/DylannDocs/blob/master/assets/MenuItems.png)

- [Basic Ideas](#basic-ideas)
- [Installation](#installation)
- [Usage & Programming](#usage)
- [Special Features](#special-features)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)

## <a name="basic-ideas"> </a> Basic Ideas

![menu-title](https://github.com/DylanWaken/DylannDocs/blob/master/assets/BasicIdeas.png)

I have attempted such frameworks in multiple ways, including multiple OOP
and POP implementations, as most of the ancestors of Dylann were abandoned due to the lack of
flexibility and architecture issues (making complex models extremely hard to code). I might be
writing a post about these learnings in the future (some blog text). For now, let's focus on how
Dylann works. 

The earliest inspiration came from assembly language:
```asm
; An assembly language example for adding and multiplying two numbers
    mov eax, 5
    mov ebx, 3
    mov ecx, 0
    add eax, ebx
    mov ecx, 1
    mul ebx
```

Assembly is fundamentally a sequence of operations, including store, load, copy, move and arithmetics. All
data are stored in an array of registers, where operations are performed. Since neural networks are
technically also sequences of matrix operations, I thought of something:

```cpp
//Instructions for a resnet block (hex stands for tensors)
CONV2D          0xa   0xb   0x9   0xc 1 1 1 1 1 1
BATCHNORM2D     0xc   0x11  0xf   0x10  0xd  0xe 1e-08 1
RELU            0x11  0x12
CONV2D          0x13  0x14  0x12  0x15 1 1 1 1 1 1
BATCHNORM2D     0x15  0x1a  0x18  0x19  0x16 0x17 1e-08 1
ADD             0x1a  0x9   0x1b  1 1
RELU            0x1b  0x1c
```

The bare form looks scary, so lets view them in graphs.

`cuTensorBase` is the representation of tensors. it includes a header sections and data section, as shown by following:

![tensor](https://github.com/DylanWaken/DylannDocs/blob/master/assets/tENSOR.png)

Each tensor object header would include the following information:
```cpp
struct TDescriptor{
        cudnnTensorDescriptor_t cudnnDesc{};   //cudnn descriptor for using the library
        
        cudnnDataType_t dType;     //tensor data type
        shape4 sizes;              //tensor shape, only supporting 4 dimension for now
        uint64_t numel;            //number of elements
        uint64_t elementSize;      //size of each element
        
        uint64_t uuid;             //unique id for the tensor
        
        //state
        bool isAllocated = false;  //whether the tensor data is allocated on device memory
        bool withGrad = false;     //whether the tensor has a gradient allocated
        bool isParam = false;      //whether the tensor is a parameter (going to be saved and optimized)
        bool isWeight = false;     //whether the tensor is a weight (been multiplies, use for L2 regularization)

        PARAM_INIT_TYPE paramInitType;  //defines how to initialize the tensor with random values
};
```
Each tensor data storgae includes the following information:
```cpp
struct TStorage{
        void* data;                //pointer to the data
        int deviceID;              //the CUDA device the tensor is on
        uint64_t memSize;          //size of data in device memory (bytes)
};
```
Each `cuTensorBase` object would include a `TDescriptor` and pointers to two `TStorage` objects. with one storage for
data and another for gradients.


![tensor-sequence](https://github.com/DylanWaken/DylannDocs/blob/master/assets/TensorSeq.png)

The engine maintains a registered array of all tensors currently in the system (like memory in normal programming). For 
safety concerns and easier management, the array is in the shape of maps, where the key is the tensor's `uuid` or its
serial number (of uint64_t type), and the values are pointers to tensor objects. All tensor definition in the framework 
will be adding a new slot to the map.
```cpp
map<uint64_t, cuTensorBase*> tensors;
        |           |
        V           V
      uuid       pointer
```

