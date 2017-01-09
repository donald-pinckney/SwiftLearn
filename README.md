# SwiftLearn
**SwiftLearn** is a collection of machine learning algorithms implemented in Swift. 
Although currently SwiftLearn does not support GPU acceleration, it uses [SwiftNum](https://github.com/donald-pinckney/SwiftNum) to be quite fast using the CPU.

## What does SwiftLearn Do?
**SwiftLearn is a clean and performance Swift interface for various machine learning algorithms.**

So far **SwiftLearn** only has feedforward neural networks implemented, but many other common machine learning algorithms are planned!

## Requirements
Currently the requirements are only that of [SwiftNum](https://github.com/donald-pinckney/SwiftNum).

## How to Use
If you want to see a demo of this working, just clone the repository, open the project with Xcode, and run the unit tests.
SwiftLearn will proceed to train a neural network on MNIST data.

To use this in a project, just add it to your `Package.swift` file:
```swift
.Package(url: "https://github.com/donald-pinckney/SwiftLearn", Version(x, y, z))
```
