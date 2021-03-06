//
//  NNLayer.swift
//  AES-NN
//
//  Created by Donald Pinckney on 12/27/16.
//
//

import Linear

public struct NeuralNetworkLayerState {
    var inputs: Matrix { // This DOES include bias input, is (inputNodeCount + 1) x 1.
        didSet {
            inputsT = inputs.T
        }
    }
    internal var inputsT: Matrix // 1 x (inputNodeCount + 1), this is just for optimization
    
    var linearOutputs: Matrix // Is outputNodeCount x 1
    var activations: Matrix // Is outputNodeCount x 1
    
    var outputErrors: Matrix // Is outputNodeCount x 1
    var backMultipliedErrors: Matrix // Is inputNodeCount x 1
    
    init(inputCount: Int, outputCount: Int) {
        inputs = Matrix.zeros(inputCount + 1, 1)
        inputsT = Matrix.zeros(1, inputCount + 1)

        linearOutputs = Matrix.zeros(outputCount, 1)
        activations = Matrix.zeros(outputCount, 1)
        outputErrors = Matrix.zeros(outputCount, 1)
        backMultipliedErrors = Matrix.zeros(inputCount, 1)
    }
}

public struct NeuralNetworkLayer {
    public let inputNodeCount: Int // Does NOT include bias node
    public let outputNodeCount: Int
    
    public let activationFunction: ActivationFunction
    
    // DOES include weights for bias node (at the left of the matrix)
    // This should be an outputCount x (inputCount + 1) dimensional matrix,
    // And we multiply like so: weights * (1 vercat inputs)
    public var weights: Matrix {
        didSet {
            weightsT = weights.T
        }
    }
    private var weightsT: Matrix // this is just for optimization
    
    public var currentState: NeuralNetworkLayerState
    
    public let constantMask: Matrix
    
    public init(weights: Matrix, activationFunction: ActivationFunction, constantMask: Matrix! = nil) {
        if constantMask == nil {
            self.constantMask = Matrix.fill(weights.height, weights.width, value: 1)
        } else {
            precondition(constantMask.width == weights.width)
            precondition(constantMask.height == weights.height)
            self.constantMask = constantMask
        }
        
        inputNodeCount = weights.width - 1
        outputNodeCount = weights.height
        self.weights = weights
        self.weightsT = weights.T
        self.activationFunction = activationFunction
        currentState = NeuralNetworkLayerState(inputCount: inputNodeCount, outputCount: outputNodeCount)
    }
    
    public init(inputNodeCount: Int, outputNodeCount: Int, activationFunction: ActivationFunction) {
        let W = 2 * Matrix.random(outputNodeCount, inputNodeCount + 1) - 1
        self.init(weights: W, activationFunction: activationFunction)
    }
    
    // inputs is an inputNodeCount x 1 matix (column vector)
    internal mutating func forwardPropagate(inputs: Matrix) {
        currentState.inputs = Matrix(rowMajorData: Array(repeating: 1.0, count: inputs.width) + inputs.data, width: inputs.width)
        currentState.linearOutputs = weights * currentState.inputs
        currentState.activations = activationFunction.evaluate(currentState.linearOutputs)
    }
    
    // Errors should be a column vector of height outputNodeCount
    internal mutating func backPropagate(finalCorrectOutputs: Matrix, exampleIndex: Int) {
        currentState.outputErrors = currentState.activations[0..<currentState.activations.height, exampleIndex] - finalCorrectOutputs
        currentState.backMultipliedErrors = weightsT * currentState.outputErrors
    }
    
    internal mutating func backPropagate(backMultipliedErrorsOfForwardLayer: Matrix, exampleIndex: Int) {
        currentState.outputErrors = backMultipliedErrorsOfForwardLayer[1..<backMultipliedErrorsOfForwardLayer.height]
            .* activationFunction.evaluateDerivative(currentState.linearOutputs[0..<currentState.linearOutputs.height, exampleIndex])
        currentState.backMultipliedErrors = weightsT * currentState.outputErrors
    }
}
