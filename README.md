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

![tensor-sequence](https://github.com/DylanWaken/DylannDocs/blob/master/assets/TensorSeq.png)

