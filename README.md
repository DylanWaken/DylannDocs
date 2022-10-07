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

I have attempted such frameworks in multiple ways, including multiple OOP
and POP implementations, as most of the ancestors of Dylann were abandoned due to the lack of
flexibility and architecture issues (making complex models extremely hard to code). I might be
writing a post about these learnings in the future (some blog text). For now, let's focus on how
Dylann works. 

